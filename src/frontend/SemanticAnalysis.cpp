// SemanticAnalysis implementation: symbol resolution, type checking and
// constant folding.  Builds the module-wide global/function symbol table and
// the type/dim info used later for subscript byte-offset lowering.
#include "SemanticAnalysis.h"

#include <cmath>
#include <cstdlib>

namespace sakura {

using namespace std;

// ---------------------------------------------------------------------------
// constant conversion / folding helpers
// ---------------------------------------------------------------------------

ConstVal convertConst(TypeCat to, const ConstVal &v) {
  if (!v.valid) return v;
  if (to == TypeCat::INT) {
    if (v.type == TypeCat::INT) return v;
    return ConstVal::intC(f2i_trunc(v.fval)); // float -> int truncates
  }
  if (to == TypeCat::FLOAT) {
    if (v.type == TypeCat::FLOAT) return v;
    return ConstVal::floatC((float)v.ival);
  }
  return v;
}

static bool truthy(const ConstVal &v) {
  return v.valid && (v.type == TypeCat::FLOAT ? v.fval != 0.0f : v.ival != 0);
}

// ---------------------------------------------------------------------------
// constant-initializer flattening (C11 initializer semantics)
// ---------------------------------------------------------------------------

namespace {

struct InitStream {
  const std::vector<std::unique_ptr<InitNode>> &items;
  size_t idx = 0;
  explicit InitStream(const std::vector<std::unique_ptr<InitNode>> &v)
      : items(v) {}
  bool empty() const { return idx >= items.size(); }
  InitNode *peek() const { return empty() ? nullptr : items[idx].get(); }
};

class Flattener {
public:
  Flattener(const std::vector<int64_t> &dims, TypeCat base, int line)
      : dims_(dims), base_(base), line_(line) {}

  // `init` may be null (=> all zero) for arrays / uninitialized globals.
  void run(InitNode *init) {
    if (dims_.empty()) {
      // scalar object
      if (!init) throw CompileError("missing initializer", line_);
      pushScalarFrom(init);
      return;
    }
    out_.clear();
    if (!init) {
      zeroAll(0);
      return;
    }
    if (init->kind != InitNode::LIST)
      throw CompileError("scalar expression initializes an array", line_);
    InitStream s(init->list);
    fillAggregate(0, s);
    if (!s.empty())
      throw CompileError("excess elements in array initializer", line_);
  }

  std::vector<ConstVal> &result() { return out_; }

private:
  const std::vector<int64_t> &dims_;
  TypeCat base_;
  int line_;
  std::vector<ConstVal> out_;

  void zeroAll(size_t level) {
    if (level == dims_.size()) {
      out_.push_back(ConstVal());
      return;
    }
    for (int64_t i = 0; i < dims_[level]; i++) zeroAll(level + 1);
  }

  // Fill the aggregate described by dims_[level..] from a brace content stream.
  // When the stream runs dry the remaining elements are zero filled.
  void fillAggregate(size_t level, InitStream &s) {
    if (level == dims_.size()) {
      // scalar element reached from a stream (only via unbraced nesting)
      consumeScalar(s);
      return;
    }
    for (int64_t i = 0; i < dims_[level]; i++) {
      InitNode *head = s.peek();
      if (head && head->kind == InitNode::LIST) {
        // braced sub-aggregate for this element: it owns the whole element
        const auto &sub = head->list;
        InitStream subS(sub);
        if (level + 1 == dims_.size()) {
          // element is a scalar: a braced scalar must contain exactly one
          // expression (possibly itself braced)
          s.idx++;
          if (subS.empty())
            throw CompileError("empty braces in initializer", line_);
          pushScalarFrom(subS.peek());
          subS.idx++;
          if (!subS.empty())
            throw CompileError("excess elements in array initializer", line_);
        } else {
          s.idx++;
          fillAggregate(level + 1, subS);
          if (!subS.empty())
            throw CompileError("excess elements in array initializer", line_);
        }
      } else {
        // either a plain scalar from the stream, or stream is dry (zero)
        if (level + 1 == dims_.size()) {
          consumeScalar(s);
        } else {
          // element is still an aggregate but no explicit brace: descend with
          // the same stream
          InitStream parent(s.items);
          parent.idx = s.idx;
          fillAggregate(level + 1, parent);
          s.idx = parent.idx;
        }
      }
    }
  }

  void consumeScalar(InitStream &s) {
    if (s.empty()) {
      out_.push_back(ConstVal()); // zero
      return;
    }
    InitNode *item = s.peek();
    if (item->kind == InitNode::LIST) {
      // a braced scalar inside a larger aggregate: must contain a single value
      const auto &sub = item->list;
      if (sub.empty())
        throw CompileError("empty braces in initializer", line_);
      if (sub.size() != 1)
        throw CompileError("excess elements in array initializer", line_);
      s.idx++;
      InitStream one(sub);
      pushScalarFrom(one.peek());
      return;
    }
    pushScalarFrom(item);
    s.idx++;
  }

  // emit the value carried by a (possibly braced) scalar initializer.
  void pushScalarFrom(InitNode *item) {
    while (item && item->kind == InitNode::LIST) {
      if (item->list.size() != 1)
        throw CompileError("invalid scalar initializer", line_);
      item = item->list[0].get();
    }
    if (!item || item->kind != InitNode::EXPR)
      throw CompileError("invalid initializer", line_);
    out_.push_back(convertConst(base_, item->expr->cval));
  }
};

} // namespace

void flattenConstInit(const std::vector<int64_t> &dims, TypeCat base,
                      InitNode *init, std::vector<ConstVal> &out, int line) {
  Flattener f(dims, base, line);
  f.run(init);
  out = std::move(f.result());
}

// ---------------------------------------------------------------------------
// SemanticAnalyzer
// ---------------------------------------------------------------------------

void SemanticAnalyzer::error(const std::string &msg, int line) {
  throw CompileError(msg, line);
}

void SemanticAnalyzer::annotateScalar(ExprNode *e, TypeCat t,
                                      const ConstVal &v) {
  e->evalType = t;
  e->isArrayExpr = false;
  e->isLValue = false;
  e->cval = v;
}

VarSymbol *SemanticAnalyzer::resolve(const std::string &name) {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    auto f = it->find(name);
    if (f != it->end()) return f->second;
  }
  return nullptr;
}

VarSymbol *SemanticAnalyzer::declare(const std::string &name, VarType type,
                                     bool isConst, bool isGlobal, bool isParam,
                                     int line) {
  if (scopes_.empty()) error("internal: no scope", line);
  auto &scope = scopes_.back();
  if (scope.count(name))
    error("redefinition of variable '" + name + "'", line);
  auto sym = std::make_unique<VarSymbol>();
  sym->name = name;
  sym->type = type;
  sym->isConst = isConst;
  sym->isGlobal = isGlobal;
  sym->isParam = isParam;
  VarSymbol *p = sym.get();
  arena_.push_back(std::move(sym));
  scope[name] = p;
  return p;
}

void SemanticAnalyzer::requireScalar(ExprNode *e) {
  if (e->isArrayExpr)
    error("array value is not usable in a scalar context", e->line);
  if (e->isTensorVal)
    error("tensor value is not usable in a scalar context", e->line);
  if (e->evalType == TypeCat::VOID)
    error("void value is not usable here", e->line);
}

void SemanticAnalyzer::requireConst(ExprNode *e, const std::string &what) {
  if (!e->cval.valid)
    error(what + " must be a compile-time constant expression", e->line);
}

// ===========================================================================
// top level
// ===========================================================================

void SemanticAnalyzer::analyze() {
  scopes_.clear();
  pushScope(); // global scope

  // globals first: symbols (with const values) must be visible to function
  // parameter dimensions such as `int graph[][V]`.
  for (auto &d : unit_->globals) analyzeGlobal(d.get());

  registerFuncSignatures();

  // Tensor-returning functions are analysed first so that their static return
  // shapes are inferred before any caller is checked against them.
  for (auto &fn : unit_->funcs)
    if (fn->retIsTensor) analyzeFuncBody(fn.get());
  for (auto &fn : unit_->funcs)
    if (!fn->retIsTensor) analyzeFuncBody(fn.get());
}

void SemanticAnalyzer::registerFuncSignatures() {
  // ps: type for scalar params; "A" = int array param, "AF" = float array
  auto addBuiltin = [&](const char *n, TypeCat ret,
                        std::initializer_list<std::string> ps) {
    FuncInfo f;
    f.ret = ret;
    f.builtin = true;
    for (auto &s : ps) {
      VarType vt;
      if (s == "i") vt.base = TypeCat::INT;
      else if (s == "f") vt.base = TypeCat::FLOAT;
      else if (s == "Ai") { vt.base = TypeCat::INT; vt.dims = {-1}; }
      else if (s == "Af") { vt.base = TypeCat::FLOAT; vt.dims = {-1}; }
      f.params.push_back(vt);
    }
    funcs_[n] = f;
  };
  addBuiltin("getint", TypeCat::INT, {});
  addBuiltin("getch", TypeCat::INT, {});
  addBuiltin("getfloat", TypeCat::FLOAT, {});
  addBuiltin("putint", TypeCat::VOID, {"i"});
  addBuiltin("putch", TypeCat::VOID, {"i"});
  addBuiltin("putfloat", TypeCat::VOID, {"f"});
  addBuiltin("getarray", TypeCat::INT, {"Ai"});
  addBuiltin("putarray", TypeCat::VOID, {"i", "Ai"});
  addBuiltin("getfarray", TypeCat::INT, {"Af"});
  addBuiltin("putfarray", TypeCat::VOID, {"i", "Af"});
  addBuiltin("starttime", TypeCat::VOID, {});
  addBuiltin("stoptime", TypeCat::VOID, {});

  for (auto &fn : unit_->funcs) {
    auto it = funcs_.find(fn->name);
    if (it != funcs_.end() && it->second.def)
      error("redefinition of function '" + fn->name + "'", fn->line);
    FuncInfo info;
    info.def = fn.get();
    info.ret = fn->retType;
    info.retIsTensor = fn->retIsTensor;
    for (auto &p : fn->params) {
      // resolve the parameter's (array) type now, against the global scope
      resolveParamType(p.get());
      info.params.push_back(p->type);
    }
    funcs_[fn->name] = std::move(info);
  }
}

void SemanticAnalyzer::resolveParamType(ParamNode *p) {
  if (p->type.isTensor && !p->isArray) {
    error("tensor parameter '" + p->name +
              "' must be declared with an omitted first dimension '[]'",
          p->line);
  }
  if (!p->isArray) return; // scalar: builder already set base
  std::vector<int64_t> dims;
  if (p->type.isTensor) {
    // A tensor parameter decays to a pointer to its first element, so its
    // first dimension is always unknown ("[]") like a plain array parameter.
    if (!p->firstDimOmitted)
      error("first dimension of tensor parameter '" + p->name +
                "' must be omitted",
            p->line);
    dims.push_back(-1);
  } else if (p->firstDimOmitted) {
    dims.push_back(-1);
  }
  for (auto &d : p->dimExprs) {
    checkExpr(d.get(), true, false);
    requireScalar(d.get());
    if (d->evalType != TypeCat::INT)
      error("array dimension must have integer type", d->line);
    requireConst(d.get(), "array dimension");
    int64_t v = d->cval.ival;
    if (v <= 0) error("array dimension must be positive", d->line);
    dims.push_back(v);
  }
  p->type.dims = std::move(dims);
}

void SemanticAnalyzer::analyzeGlobal(DeclNode *decl) {
  TypeCat base = decl->base;
  for (auto &def : decl->defs)
    checkGlobalDef(def.get(), decl->isConst, base, decl->isTensor);
}

void SemanticAnalyzer::checkGlobalDef(VarDefNode *def, bool isConst,
                                      TypeCat base, bool isTensor) {
  vector<int64_t> dims;
  for (auto &d : def->dims) {
    checkExpr(d.get(), true, false);
    requireScalar(d.get());
    if (d->evalType != TypeCat::INT)
      error("array dimension must have integer type", d->line);
    requireConst(d.get(), "array dimension");
    int64_t v = d->cval.ival;
    if (v <= 0) error("array dimension must be positive", d->line);
    dims.push_back(v);
  }
  if (isTensor && dims.empty())
    error("tensor variable '" + def->name +
              "' must be declared with at least one dimension",
          def->line);
  VarType t;
  t.base = base;
  t.isConst = isConst;
  t.isTensor = isTensor;
  t.dims = dims;
  def->type = t;

  // const / global initializers must be compile-time constants
  if (def->init) checkConstInitTree(base, def->init.get(), def->line);

  VarSymbol *sym = declare(def->name, t, isConst, true, false, def->line);
  sym->hasInit = def->init != nullptr;

  if (dims.empty()) {
    if (isConst) {
      if (!def->init)
        error("const scalar '" + def->name + "' needs an initializer", def->line);
      sym->value = convertConst(base, def->init->expr->cval);
    } else if (def->init) {
      sym->value = convertConst(base, def->init->expr->cval);
    }
  } else {
    if (def->init) {
      flattenConstInit(dims, base, def->init.get(), sym->flat, def->line);
      sym->hasFlat = true;
    }
  }
}

void SemanticAnalyzer::checkConstInitTree(TypeCat base, InitNode *init,
                                          int line) {
  if (init->kind == InitNode::EXPR) {
    checkExpr(init->expr.get(), true, false);
    requireScalar(init->expr.get());
    requireConst(init->expr.get(), "initializer");
    init->expr->cval = convertConst(base, init->expr->cval);
  } else {
    for (auto &ch : init->list) checkConstInitTree(base, ch.get(), line);
  }
}

void SemanticAnalyzer::checkLocalDef(VarDefNode *def, bool isConst,
                                     TypeCat base, bool isTensor) {
  vector<int64_t> dims;
  for (auto &d : def->dims) {
    checkExpr(d.get(), true, false);
    requireScalar(d.get());
    if (d->evalType != TypeCat::INT)
      error("array dimension must have integer type", d->line);
    requireConst(d.get(), "array dimension");
    int64_t v = d->cval.ival;
    if (v <= 0) error("array dimension must be positive", d->line);
    dims.push_back(v);
  }
  if (isTensor && dims.empty())
    error("tensor variable '" + def->name +
              "' must be declared with at least one dimension",
          def->line);
  VarType t;
  t.base = base;
  t.isConst = isConst;
  t.isTensor = isTensor;
  t.dims = dims;
  def->type = t;

  VarSymbol *sym = declare(def->name, t, isConst, false, false, def->line);
  sym->hasInit = def->init != nullptr;

  if (dims.empty()) {
    if (isConst) {
      if (!def->init)
        error("const scalar '" + def->name + "' needs an initializer", def->line);
      if (def->init->kind != InitNode::EXPR)
        error("scalar initializer must be an expression", def->line);
      checkExpr(def->init->expr.get(), true, false);
      requireScalar(def->init->expr.get());
      requireConst(def->init->expr.get(), "initializer");
      sym->value = convertConst(base, def->init->expr->cval);
    } else if (def->init) {
      if (def->init->kind != InitNode::EXPR)
        error("scalar initializer must be an expression", def->line);
      checkExpr(def->init->expr.get(), true, false);
      requireScalar(def->init->expr.get());
    }
  } else {
    if (def->init) {
      if (isConst) {
        checkConstInitTree(base, def->init.get(), def->line);
        flattenConstInit(dims, base, def->init.get(), sym->flat, def->line);
        sym->hasFlat = true;
      } else {
        checkLocalInitTree(base, def->init.get(), def->line);
      }
    }
  }
}

void SemanticAnalyzer::checkLocalInitTree(TypeCat base, InitNode *init,
                                          int line) {
  if (init->kind == InitNode::EXPR) {
    checkExpr(init->expr.get(), true, false);
    requireScalar(init->expr.get());
  } else {
    for (auto &ch : init->list) checkLocalInitTree(base, ch.get(), line);
  }
}

void SemanticAnalyzer::analyzeFuncBody(FuncDefNode *fn) {
  auto it = funcs_.find(fn->name);
  if (it == funcs_.end()) {
    FuncInfo info;
    info.def = fn;
    info.ret = fn->retType;
    info.retIsTensor = fn->retIsTensor;
    for (auto &p : fn->params) info.params.push_back(p->type);
    it = funcs_.insert({fn->name, std::move(info)}).first;
  }
  curFunc_ = &it->second;
  curFnNode_ = fn;

  pushScope();
  for (auto &p : fn->params) {
    VarSymbol *sym = declare(p->name, p->type, false, false, true, p->line);
    sym->hasInit = true;
  }
  loopDepth_ = 0;
  checkBlock(fn->body.get());
  popScope();
  if (fn->retIsTensor && fn->retDims.empty())
    error("tensor function '" + fn->name +
              "' must return a whole tensor of statically known shape",
          fn->line);
  curFunc_ = nullptr;
  curFnNode_ = nullptr;
}

// ===========================================================================
// statements
// ===========================================================================

void SemanticAnalyzer::checkBlock(BlockStmtNode *b) {
  pushScope();
  for (auto &s : b->body) checkStmt(s.get());
  popScope();
}

void SemanticAnalyzer::checkStmt(StmtNode *s) {
  if (auto *ds = dynamic_cast<DeclStmtNode *>(s)) {
    TypeCat base = ds->decl->base;
    for (auto &def : ds->decl->defs)
      checkLocalDef(def.get(), ds->decl->isConst, base, ds->decl->isTensor);
  } else if (auto *bs = dynamic_cast<BlockStmtNode *>(s)) {
    checkBlock(bs);
  } else if (auto *as = dynamic_cast<AssignStmtNode *>(s)) {
    checkAssign(as);
  } else if (auto *es = dynamic_cast<ExprStmtNode *>(s)) {
    if (es->expr) {
      checkExpr(es->expr.get(), true, true);
      if (es->expr->isArrayExpr)
        error("array value used as a statement", es->expr->line);
      if (es->expr->isTensorVal)
        error("tensor value used as a statement", es->expr->line);
    }
  } else if (auto *is = dynamic_cast<IfStmtNode *>(s)) {
    checkCond(is->cond.get());
    checkStmt(is->thenStmt.get());
    if (is->elseStmt) checkStmt(is->elseStmt.get());
  } else if (auto *ws = dynamic_cast<WhileStmtNode *>(s)) {
    checkCond(ws->cond.get());
    loopDepth_++;
    checkStmt(ws->body.get());
    loopDepth_--;
  } else if (auto *br = dynamic_cast<BreakStmtNode *>(s)) {
    if (loopDepth_ == 0) error("'break' outside of a loop", br->line);
  } else if (auto *ct = dynamic_cast<ContinueStmtNode *>(s)) {
    if (loopDepth_ == 0) error("'continue' outside of a loop", ct->line);
  } else if (auto *rt = dynamic_cast<ReturnStmtNode *>(s)) {
    checkReturn(rt);
  }
}

void SemanticAnalyzer::checkCond(ExprNode *e) {
  checkExpr(e, true, false);
  requireScalar(e);
}

void SemanticAnalyzer::checkAssign(AssignStmtNode *s) {
  LValNode *lv = dynamic_cast<LValNode *>(s->lhs.get());
  if (!lv) error("left side of assignment is not an lvalue", s->line);
  checkLValExpr(lv, false);
  if (lv->varIsConst) error("cannot assign to a const variable", s->line);
  if (lv->isTensorVal) {
    // whole-tensor assignment: `t = expr` with identical shapes/element types
    checkExpr(s->rhs.get(), true, false);
    if (!s->rhs->isTensorVal)
      error("a tensor can only be assigned a whole tensor value", s->line);
    if (s->rhs->evalType != lv->evalType)
      error("tensor element type mismatch in assignment", s->line);
    for (int64_t d : lv->varDims)
      if (d < 1)
        error("cannot assign to a tensor with an unknown dimension", s->line);
    for (int64_t d : s->rhs->tShape)
      if (d < 1)
        error("cannot assign a tensor of unknown shape", s->line);
    if (s->rhs->tShape != lv->varDims)
      error("tensor shape mismatch in assignment", s->line);
    return;
  }
  if (lv->isArrayExpr) error("cannot assign to an array", s->line);
  if (lv->evalType == TypeCat::VOID)
    error("cannot assign to a void expression", s->line);
  checkExpr(s->rhs.get(), true, false);
  requireScalar(s->rhs.get());
}

void SemanticAnalyzer::checkReturn(ReturnStmtNode *s) {
  if (curFunc_->retIsTensor) {
    if (!s->value) error("tensor function must return a tensor value", s->line);
    checkExpr(s->value.get(), true, false);
    if (!s->value->isTensorVal)
      error("tensor function must return a whole tensor", s->line);
    if (s->value->evalType != curFunc_->ret)
      error("returned tensor element type does not match the function type",
            s->line);
    for (int64_t d : s->value->tShape)
      if (d < 1)
        error("returned tensor must have a statically known shape", s->line);
    if (curFunc_->retDims.empty()) {
      curFunc_->retDims = s->value->tShape;
      curFnNode_->retDims = s->value->tShape;
    } else if (curFunc_->retDims != s->value->tShape) {
      error("all return statements of a tensor function must return tensors "
            "of the same shape",
            s->line);
    }
    return;
  }
  if (curFunc_->ret == TypeCat::VOID) {
    if (s->value) error("void function cannot return a value", s->line);
    return;
  }
  if (!s->value) error("non-void function must return a value", s->line);
  checkExpr(s->value.get(), true, false);
  requireScalar(s->value.get());
}

// ===========================================================================
// expressions
// ===========================================================================

void SemanticAnalyzer::checkExpr(ExprNode *e, bool asRvalue,
                                 bool allowArrayResult) {
  if (auto *lit = dynamic_cast<IntLitNode *>(e)) {
    annotateScalar(lit, TypeCat::INT, ConstVal::intC(lit->value));
  } else if (auto *fl = dynamic_cast<FloatLitNode *>(e)) {
    float v = strtof(fl->text.c_str(), nullptr);
    annotateScalar(fl, TypeCat::FLOAT, ConstVal::floatC(v));
  } else if (auto *sl = dynamic_cast<StringLitNode *>(e)) {
    (void)sl;
    e->evalType = TypeCat::VOID;
    e->isArrayExpr = false;
    e->isLValue = false;
  } else if (auto *lv = dynamic_cast<LValNode *>(e)) {
    checkLValExpr(lv, asRvalue);
    if (lv->isArrayExpr && !allowArrayResult && asRvalue)
      error("array expression used as a value", lv->line);
  } else if (auto *cl = dynamic_cast<CallExprNode *>(e)) {
    checkCall(cl);
  } else if (auto *un = dynamic_cast<UnaryExprNode *>(e)) {
    checkUnary(un);
  } else if (auto *bn = dynamic_cast<BinaryExprNode *>(e)) {
    checkBinary(bn);
  }
}

void SemanticAnalyzer::checkLValExpr(LValNode *e, bool asRvalue) {
  VarSymbol *sym = resolve(e->name);
  if (!sym) error("undefined variable '" + e->name + "'", e->line);

  for (auto &idx : e->indices) {
    checkExpr(idx.get(), true, false);
    requireScalar(idx.get());
    if (idx->evalType != TypeCat::INT)
      error("array subscript must have integer type", idx->line);
  }

  const vector<int64_t> &dims = sym->type.dims;
  e->varDims = dims;
  e->varKind =
      sym->isGlobal ? LValKind::GLOBAL_VAR
                    : (sym->isParam ? LValKind::PARAM : LValKind::LOCAL_VAR);
  e->varIsConst = sym->isConst;
  e->varIsTensor = sym->type.isTensor;
  e->evalType = sym->type.base;

  if (e->varIsTensor) {
    // A tensor object is only addressable as a whole or through a full set of
    // subscripts (one scalar element); partial subscripting is not allowed.
    if (e->indices.size() == 0) {
      e->isTensorVal = true; // whole tensor value
      e->isArrayExpr = false;
      e->isLValue = true;
      e->tShape = dims;
    } else if (e->indices.size() == dims.size()) {
      e->isTensorVal = false; // a single scalar element
      e->isArrayExpr = false;
      e->isLValue = true;
    } else {
      error("tensor '" + e->name + "' needs exactly " +
                to_string(dims.size()) + " subscripts (no partial access)",
            e->line);
    }
  } else if (dims.empty()) {
    if (!e->indices.empty())
      error("too many subscripts for scalar '" + e->name + "'", e->line);
    e->isArrayExpr = false;
    e->isLValue = true;
  } else {
    if (e->indices.size() > dims.size())
      error("too many subscripts for '" + e->name + "'", e->line);
    e->isArrayExpr = e->indices.size() < dims.size();
    e->isLValue = true;
  }

  if (sym->isConst && asRvalue) {
    if (sym->type.dims.empty() && sym->hasInit) {
      e->cval = sym->value;
    } else if (!sym->type.dims.empty() && !e->isArrayExpr && !e->isTensorVal &&
               e->indices.size() == sym->type.dims.size() && sym->hasFlat) {
      bool allConst = true;
      for (auto &idx : e->indices)
        if (!idx->cval.valid || idx->cval.type != TypeCat::INT) allConst = false;
      if (allConst) {
        int64_t off = 0;
        for (size_t i = 0; i < e->indices.size(); i++) {
          off = off * dims[i] + e->indices[i]->cval.ival;
        }
        if (off >= 0 && (size_t)off < sym->flat.size())
          e->cval = sym->flat[(size_t)off];
      }
    }
  }
}

void SemanticAnalyzer::checkCall(CallExprNode *e) {
  auto it = funcs_.find(e->name);
  if (it == funcs_.end())
    error("call to undeclared function '" + e->name + "'", e->line);
  FuncInfo &fi = it->second;
  if (e->args.size() != fi.params.size()) {
    error("function '" + e->name + "' expects " + to_string(fi.params.size()) +
              " argument(s), got " + to_string(e->args.size()),
          e->line);
  }
  for (size_t i = 0; i < e->args.size(); i++) {
    VarType &pt = fi.params[i];
    if (pt.isTensor) {
      // tensor formal: expects a *whole tensor* of matching element type and
      // shape (first dimension of the formal is "[]" = wildcard).
      checkExpr(e->args[i].get(), true, false);
      if (!e->args[i]->isTensorVal)
        error("argument " + to_string(i + 1) + " of '" + e->name +
                  "' must be a whole tensor",
              e->args[i]->line);
      if (e->args[i]->evalType != pt.base)
        error("tensor element type mismatch in argument " +
                  to_string(i + 1) + " of '" + e->name + "'",
              e->args[i]->line);
      if (!tensorShapeMatches(pt, e->args[i].get()))
        error("tensor shape mismatch in argument " + to_string(i + 1) +
                  " of '" + e->name + "'",
              e->args[i]->line);
    } else if (!pt.dims.empty()) {
      // plain SysY array formal
      checkExpr(e->args[i].get(), true, true);
      if (!e->args[i]->isArrayExpr)
        error("argument " + to_string(i + 1) + " of '" + e->name +
                  "' must be an array",
              e->args[i]->line);
      if (e->args[i]->evalType != pt.base)
        error("array element type mismatch in argument " + to_string(i + 1) +
                  " of '" + e->name + "'",
              e->args[i]->line);
    } else {
      checkExpr(e->args[i].get(), true, false);
      requireScalar(e->args[i].get());
    }
  }
  e->evalType = fi.ret;
  e->isArrayExpr = false;
  e->isLValue = false;
  if (fi.retIsTensor) {
    // Whole-tensor result; the static shape comes from the callee signature
    // (inferred during semantic analysis of the callee body).
    e->isTensorVal = true;
    e->tShape = fi.retDims;
  }
}

void SemanticAnalyzer::checkUnary(UnaryExprNode *e) {
  checkExpr(e->operand.get(), true, false);
  if (e->operand->isTensorVal) {
    checkTensorUnary(e);
    return;
  }
  requireScalar(e->operand.get());
  TypeCat ot = e->operand->evalType;
  e->evalType = (e->op == UnaryOp::NOT) ? TypeCat::INT : ot;
  e->isArrayExpr = false;
  e->isLValue = false;
  foldUnary(e);
}

void SemanticAnalyzer::checkTensorUnary(UnaryExprNode *e) {
  if (e->op == UnaryOp::NOT)
    error("'!' is not applicable to a tensor operand", e->line);
  e->evalType = e->operand->evalType;
  e->isArrayExpr = false;
  e->isLValue = false;
  e->isTensorVal = true;
  e->tShape = e->operand->tShape;
}

void SemanticAnalyzer::checkBinary(BinaryExprNode *e) {
  checkExpr(e->lhs.get(), true, false);
  checkExpr(e->rhs.get(), true, false);
  if (e->lhs->isTensorVal || e->rhs->isTensorVal) {
    checkTensorBinary(e);
    return;
  }
  requireScalar(e->lhs.get());
  requireScalar(e->rhs.get());

  BinaryOp op = e->op;
  if (op == BinaryOp::AND || op == BinaryOp::OR) {
    e->evalType = TypeCat::INT;
    e->isArrayExpr = false;
    e->isLValue = false;
    foldBinary(e);
    return;
  }
  TypeCat lt = e->lhs->evalType, rt = e->rhs->evalType;
  if (lt == TypeCat::VOID || rt == TypeCat::VOID)
    error("void value in expression", e->line);
  if (op == BinaryOp::MOD && (lt == TypeCat::FLOAT || rt == TypeCat::FLOAT))
    error("'%' requires integer operands", e->line);

  switch (op) {
  case BinaryOp::LT:
  case BinaryOp::GT:
  case BinaryOp::LE:
  case BinaryOp::GE:
  case BinaryOp::EQ:
  case BinaryOp::NE:
    e->evalType = TypeCat::INT;
    break;
  default:
    e->evalType = (lt == TypeCat::FLOAT || rt == TypeCat::FLOAT)
                      ? TypeCat::FLOAT
                      : TypeCat::INT;
    break;
  }
  e->isArrayExpr = false;
  e->isLValue = false;
  foldBinary(e);
}

void SemanticAnalyzer::setTensorResult(ExprNode *e, TypeCat base,
                                       const std::vector<int64_t> &shape) {
  e->evalType = base;
  e->isArrayExpr = false;
  e->isLValue = false;
  e->isTensorVal = true;
  e->tShape = shape;
}

// True when the formal tensor parameter `param` (dims start with -1, i.e.
// "[]") accepts the whole-tensor value `arg`.
bool SemanticAnalyzer::tensorShapeMatches(const VarType &param,
                                          const ExprNode *arg) const {
  const auto &p = param.dims;
  const auto &a = arg->tShape;
  if (p.size() != a.size()) return false;
  if (a.empty() || a[0] < 1) return false; // arg's first dim must be known
  for (size_t i = 1; i < p.size(); i++)
    if (p[i] != a[i]) return false;
  return true;
}

// Element-wise arithmetic on tensors following the scalar-promotion rules:
//   tensor op scalar / scalar op tensor : scalar is broadcast to every
//   element, tensor op tensor requires identical shapes, `@` is matrix
//   multiplication (rank 2 x rank 2).
void SemanticAnalyzer::checkTensorBinary(BinaryExprNode *e) {
  ExprNode *l = e->lhs.get(), *r = e->rhs.get();
  BinaryOp op = e->op;

  switch (op) {
  case BinaryOp::AND:
  case BinaryOp::OR:
  case BinaryOp::LT:
  case BinaryOp::GT:
  case BinaryOp::LE:
  case BinaryOp::GE:
  case BinaryOp::EQ:
  case BinaryOp::NE:
    error("relation / logical operators do not accept tensor operands",
          e->line);
    return;
  default:
    break;
  }

  if (op == BinaryOp::MATMUL) {
    if (!l->isTensorVal || !r->isTensorVal)
      error("'@' (matrix multiply) requires two tensor operands", e->line);
    if (l->evalType != r->evalType || l->evalType == TypeCat::VOID)
      error("matrix operand element types must match", e->line);
    if (l->tShape.size() != 2 || r->tShape.size() != 2)
      error("'@' requires rank-2 tensor operands (MxN @ NxP)", e->line);
    for (int64_t d : l->tShape)
      if (d < 1)
        error("'@' operand must have a statically known shape", e->line);
    for (int64_t d : r->tShape)
      if (d < 1)
        error("'@' operand must have a statically known shape", e->line);
    if (l->tShape[1] != r->tShape[0])
      error("matrix dimensions do not agree for '@' (" +
                to_string(l->tShape[0]) + "x" + to_string(l->tShape[1]) +
                " @ " + to_string(r->tShape[0]) + "x" +
                to_string(r->tShape[1]) + ")",
            e->line);
    setTensorResult(e, l->evalType, {l->tShape[0], r->tShape[1]});
    return;
  }

  // element-wise operations with scalar promotion
  TypeCat elemTy;
  std::vector<int64_t> shape;
  if (l->isTensorVal && r->isTensorVal) {
    if (l->evalType != r->evalType || l->evalType == TypeCat::VOID)
      error("tensor operand element types must match", e->line);
    if (l->tShape != r->tShape)
      error("tensor operands must have identical shapes", e->line);
    elemTy = l->evalType;
    shape = l->tShape;
  } else {
    // exactly one tensor operand, one scalar: broadcast the scalar
    ExprNode *t = l->isTensorVal ? l : r;
    ExprNode *s = l->isTensorVal ? r : l;
    requireScalar(s);
    if (s->evalType != t->evalType)
      error("scalar operand type must match the tensor element type", e->line);
    elemTy = t->evalType;
    shape = t->tShape;
  }
  if (op == BinaryOp::MOD && elemTy == TypeCat::FLOAT)
    error("'%' requires integer tensor operands", e->line);
  for (int64_t d : shape)
    if (d < 1)
      error("tensor operation requires a statically known shape", e->line);
  setTensorResult(e, elemTy, shape);
}

// ===========================================================================
// constant folding
// ===========================================================================

static float toF(const ConstVal &v) {
  return v.type == TypeCat::FLOAT ? v.fval : (float)v.ival;
}

void SemanticAnalyzer::foldUnary(UnaryExprNode *e) {
  ExprNode *o = e->operand.get();
  if (!o->cval.valid) return;
  const ConstVal &c = o->cval;
  if (e->op == UnaryOp::NOT) {
    e->cval = ConstVal::intC(truthy(c) ? 0 : 1);
  } else if (c.type == TypeCat::FLOAT) {
    e->cval =
        ConstVal::floatC(f32(e->op == UnaryOp::MINUS ? -c.fval : c.fval));
  } else {
    int64_t v = (e->op == UnaryOp::MINUS ? -c.ival : c.ival);
    e->cval = ConstVal::intC(v);
  }
}

void SemanticAnalyzer::foldBinary(BinaryExprNode *e) {
  ExprNode *l = e->lhs.get(), *r = e->rhs.get();
  if (!l->cval.valid || !r->cval.valid) return;
  ConstVal &a = l->cval;
  ConstVal &b = r->cval;
  BinaryOp op = e->op;
  bool floats = (a.type == TypeCat::FLOAT || b.type == TypeCat::FLOAT);

  switch (op) {
  case BinaryOp::AND:
    e->cval = ConstVal::intC(truthy(a) && truthy(b) ? 1 : 0);
    return;
  case BinaryOp::OR:
    e->cval = ConstVal::intC(truthy(a) || truthy(b) ? 1 : 0);
    return;
  case BinaryOp::LT:
  case BinaryOp::GT:
  case BinaryOp::LE:
  case BinaryOp::GE:
  case BinaryOp::EQ:
  case BinaryOp::NE: {
    int res;
    if (floats) {
      float x = toF(a), y = toF(b);
      switch (op) {
      case BinaryOp::LT: res = x < y; break;
      case BinaryOp::GT: res = x > y; break;
      case BinaryOp::LE: res = x <= y; break;
      case BinaryOp::GE: res = x >= y; break;
      case BinaryOp::EQ: res = x == y; break;
      default: res = x != y; break;
      }
    } else {
      int32_t x = (int32_t)a.ival, y = (int32_t)b.ival;
      switch (op) {
      case BinaryOp::LT: res = x < y; break;
      case BinaryOp::GT: res = x > y; break;
      case BinaryOp::LE: res = x <= y; break;
      case BinaryOp::GE: res = x >= y; break;
      case BinaryOp::EQ: res = x == y; break;
      default: res = x != y; break;
      }
    }
    e->cval = ConstVal::intC(res);
    return;
  }
  case BinaryOp::MOD: {
    int64_t dv = (int32_t)b.ival;
    if (dv == 0) error("division by zero in constant expression", e->line);
    e->cval = ConstVal::intC((int32_t)a.ival % dv);
    return;
  }
  case BinaryOp::DIV: {
    if (floats) {
      float y = toF(b);
      if (y == 0.0f) error("division by zero in constant expression", e->line);
      e->cval = ConstVal::floatC(f32(toF(a) / y));
    } else {
      int64_t dv = (int32_t)b.ival;
      if (dv == 0) error("division by zero in constant expression", e->line);
      e->cval = ConstVal::intC((int32_t)a.ival / dv);
    }
    return;
  }
  default:
    break;
  }

  if (floats) {
    float x = toF(a), y = toF(b);
    switch (op) {
    case BinaryOp::ADD: e->cval = ConstVal::floatC(f32(x + y)); break;
    case BinaryOp::SUB: e->cval = ConstVal::floatC(f32(x - y)); break;
    case BinaryOp::MUL: e->cval = ConstVal::floatC(f32(x * y)); break;
    default: break;
    }
  } else {
    int32_t x = (int32_t)a.ival, y = (int32_t)b.ival;
    switch (op) {
    case BinaryOp::ADD: e->cval = ConstVal::intC((int64_t)x + y); break;
    case BinaryOp::SUB: e->cval = ConstVal::intC((int64_t)x - y); break;
    case BinaryOp::MUL: e->cval = ConstVal::intC((int64_t)x * y); break;
    default: break;
    }
  }
}

} // namespace sakura
