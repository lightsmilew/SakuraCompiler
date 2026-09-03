// IRBuilder implementation: lowers the checked AST to the MLIR-style mid-end IR.
#include "IRBuilder.h"

#include <algorithm>

namespace sakura {
namespace ir {

using namespace std;

// ---------------------------------------------------------------------------
// scopes
// ---------------------------------------------------------------------------

void IRBuilder::popScope() {
  if (scopeStack_.empty()) return;
  ScopeFrame &f = scopeStack_.back();
  for (auto &kv : f.saved) {
    if (kv.second) env_[kv.first] = kv.second;
    else env_.erase(kv.first);
  }
  scopeStack_.pop_back();
}

void IRBuilder::bind(const string &name, Obj *o) {
  auto it = env_.find(name);
  Obj *old = (it != env_.end()) ? it->second : nullptr;
  scopeStack_.back().saved.emplace_back(name, old);
  env_[name] = o;
}

IRBuilder::Obj *IRBuilder::lookup(const string &name) {
  auto it = env_.find(name);
  return it == env_.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------------------
// IR emission primitives
// ---------------------------------------------------------------------------

BasicBlock *IRBuilder::newBlock() {
  auto b = make_unique<BasicBlock>("label" + to_string(blockSeq_++));
  BasicBlock *p = b.get();
  regStack_.back()->push_back(move(b));
  return p;
}

Instruction *IRBuilder::newInstr(Op op, Type resultTy, int line) {
  auto inst = make_unique<Instruction>(op, resultTy);
  inst->line = line;
  Instruction *p = inst.get();
  if (p->hasResult()) p->name = "t" + to_string(tempSeq_++);
  curBB_->instrs.push_back(move(inst));
  return p;
}

void IRBuilder::emitTerm(Op op, const vector<Value *> &ops, int line) {
  Instruction *i = newInstr(op, Type::Void, line);
  i->ops = ops;
}

void IRBuilder::emitStore(Value *addr, Value *val, int line) {
  Instruction *i = newInstr(Op::Store, Type::Void, line);
  i->ops = {val, addr};
}

Value *IRBuilder::emitLoad(Value *addr, Type elem, int line) {
  Instruction *i = newInstr(Op::Load, elem, line);
  i->ops = {addr};
  return i;
}

Value *IRBuilder::emitAlloca(Type elem, int64_t count, int line) {
  Instruction *i = newInstr(Op::Alloca, Type::Ptr, line);
  i->elem = elem;
  i->n = count;
  return i;
}

Value *IRBuilder::emitGep(Value *base, Value *byteOff, int line) {
  if (auto *c = dynamic_cast<ConstantInt *>(byteOff)) {
    if (c->v == 0) return base;
  }
  Instruction *i = newInstr(Op::Gep, Type::Ptr, line);
  i->ops = {base, byteOff};
  return i;
}

Value *IRBuilder::convert(Value *v, Type dst, int line) {
  if (v->ty == dst) return v;
  if (auto *ci = dynamic_cast<ConstantInt *>(v)) {
    if (dst == Type::F32) return cF((float)ci->v);
  } else if (auto *cf = dynamic_cast<ConstantFloat *>(v)) {
    if (dst == Type::I32) return cI(f2i_trunc(cf->v));
  }
  Op op = (dst == Type::F32) ? Op::Sitofp : Op::Fptosi;
  Instruction *i = newInstr(op, dst, line);
  i->ops = {v};
  return i;
}

Value *IRBuilder::constOf(const ConstVal &cv) {
  if (cv.type == TypeCat::FLOAT) return cF(cv.fval);
  return cI((int32_t)cv.ival);
}

Value *IRBuilder::emitCmp(Cond c, Value *l, Value *r, bool isFloat, int line) {
  Instruction *i = newInstr(isFloat ? Op::FCmp : Op::ICmp, Type::I32, line);
  i->cond = c;
  i->ops = {l, r};
  return i;
}

// ---------------------------------------------------------------------------
// module passes
// ---------------------------------------------------------------------------

unique_ptr<Module> IRBuilder::run() {
  pushScope(); // global scope, lives for the whole module
  declareFunctions();
  emitGlobals();
  for (auto &fn : unit_->funcs) buildFunction(fn.get());
  popScope();
  return move(mod_);
}

void IRBuilder::declareFunctions() {
  for (auto &fn : unit_->funcs) {
    if (mod_->getFunction(fn->name)) continue;
    vector<ParamDesc> ps;
    // A tensor-returning function is lowered to a function that writes its
    // result into a caller-provided buffer (the "sret" parameter), so its IR
    // signature gets an extra leading pointer argument.
    if (fn->retIsTensor) ps.push_back(ParamDesc{Type::Ptr, true, ".tret"});
    for (auto &p : fn->params) {
      ParamDesc d;
      d.name = p->name;
      d.isArray = p->isArray;
      d.ty = p->isArray ? Type::Ptr : typeOf(p->type.base);
      ps.push_back(d);
    }
    mod_->addFunction(fn->name,
                      fn->retIsTensor ? Type::Void : typeOf(fn->retType), ps,
                      false);
  }
}

Function *IRBuilder::findOrDeclareFunc(const string &name) {
  if (Function *f = mod_->getFunction(name)) return f;
  auto it = funcs_->find(name);
  if (it == funcs_->end())
    throw CompileError("internal: call to undeclared function '" + name + "'");
  const FuncInfo &fi = it->second;
  // The perf-test runtime exports the timers under their underscored names;
  // SysY sources only ever call the friendly starttime()/stoptime() forms.
  string sym = name;
  if (fi.builtin && name == "starttime") sym = "_sysy_starttime";
  if (fi.builtin && name == "stoptime") sym = "_sysy_stoptime";
  vector<ParamDesc> ps;
  for (auto &vt : fi.params) {
    ParamDesc d;
    d.isArray = !vt.dims.empty();
    d.ty = d.isArray ? Type::Ptr : typeOf(vt.base);
    ps.push_back(d);
  }
  return mod_->addFunction(sym, typeOf(fi.ret), ps, true);
}

void IRBuilder::emitGlobals() {
  for (auto &decl : unit_->globals) {
    for (auto &def : decl->defs) emitGlobalDef(def.get(), decl->base, decl->isConst);
  }
}

void IRBuilder::emitGlobalDef(VarDefNode *def, TypeCat base, bool isConst) {
  const VarType &t = def->type;
  Type elem = typeOf(base);
  vector<ConstVal> flat;
  bool hasInit = def->init != nullptr;

  if (t.dims.empty()) {
    if (isConst) return; // reads are folded constants; no storage needed
    flat.assign(1, ConstVal());
    if (def->init) flat[0] = convertConst(base, def->init->expr->cval);
    auto *g = mod_->addGlobal(def->name, elem, 1, isConst, hasInit, flat);
    auto *o = new Obj;
    o->addr = g;
    o->elem = elem;
    o->isConst = isConst;
    bind(def->name, o);
    return;
  }

  int64_t total = t.numElements();
  flat.assign((size_t)total, ConstVal());
  if (hasInit) flattenConstInit(t.dims, base, def->init.get(), flat, def->line);
  auto *g = mod_->addGlobal(def->name, elem, total, isConst, hasInit, flat);
  auto *o = new Obj;
  o->addr = g;
  o->elem = elem;
  o->isConst = isConst;
  o->isArray = true;
  o->dims = t.dims;
  bind(def->name, o);
}

void IRBuilder::buildFunction(FuncDefNode *fnNode) {
  Function *fn = mod_->getFunction(fnNode->name);
  curFn_ = fn;
  retIsTensorFn_ = fnNode->retIsTensor;
  retBuf_ = nullptr;
  tempSeq_ = 0;
  blockSeq_ = 0;
  regStack_.clear();
  regStack_.push_back(&fn->blocks); // new blocks land at function level

  pushScope(); // parameter scope
  BasicBlock *entry = newBlock();
  entry->name = "entry";
  curBB_ = entry;

  size_t a0 = 0; // index of the first source parameter in fn->args
  if (retIsTensorFn_) {
    retBuf_ = fn->args[0].get(); // hidden sret buffer
    a0 = 1;
  }
  for (size_t i = 0; i < fnNode->params.size(); ++i) {
    ParamDesc &p = fn->params[i + a0];
    Argument *arg = fn->args[i + a0].get();
    auto *o = new Obj;
    if (p.isArray) {
      o->addr = arg;
      o->elem = fnNode->params[i]->type.base == TypeCat::FLOAT ? Type::F32
                                                              : Type::I32;
      o->isArray = true;
      o->isConst = false;
      o->dims = fnNode->params[i]->type.dims;
    } else {
      Value *slot = emitAlloca(p.ty, 1, fnNode->line);
      o->addr = slot;
      o->elem = p.ty;
      o->isArray = false;
      o->isConst = false;
      emitStore(slot, arg, fnNode->line);
    }
    bind(p.name, o);
  }

  pushScope(); // function body scope
  visitBlock(fnNode->body.get());
  popScope();
  finishFunction();
  popScope();

  curFn_ = nullptr;
  curBB_ = nullptr;
  retBuf_ = nullptr;
  retIsTensorFn_ = false;
}

void IRBuilder::finishFunction() {
  // Any block without a terminator is unreachable or falls off the function;
  // close it with a default return.
  for (auto &b : curFn_->blocks) {
    if (!b->hasTerminator()) {
      curBB_ = b.get();
      Instruction *i = newInstr(Op::Ret, Type::Void, 0);
      if (curFn_->ret == Type::F32) i->ops = {cF(0.0f)};
      else if (curFn_->ret == Type::I32) i->ops = {cI(0)};
    }
  }
}

// ---------------------------------------------------------------------------
// statements
// ---------------------------------------------------------------------------

void IRBuilder::visitStmt(StmtNode *s) {
  if (curBB_->hasTerminator()) curBB_ = newBlock();
  if (auto *blk = dynamic_cast<BlockStmtNode *>(s)) {
    visitBlock(blk);
  } else if (auto *ds = dynamic_cast<DeclStmtNode *>(s)) {
    visitDeclStmt(ds);
  } else if (auto *as = dynamic_cast<AssignStmtNode *>(s)) {
    visitAssign(as);
  } else if (auto *es = dynamic_cast<ExprStmtNode *>(s)) {
    visitExprStmt(es);
  } else if (auto *is = dynamic_cast<IfStmtNode *>(s)) {
    visitIf(is);
  } else if (auto *ws = dynamic_cast<WhileStmtNode *>(s)) {
    visitWhile(ws);
  } else if (auto *bs = dynamic_cast<BreakStmtNode *>(s)) {
    (void)bs;
    emitTerm(Op::Br, {loops_.back().end}, s->line);
  } else if (auto *cs = dynamic_cast<ContinueStmtNode *>(s)) {
    (void)cs;
    emitTerm(Op::Br, {loops_.back().cond}, s->line);
  } else if (auto *rs = dynamic_cast<ReturnStmtNode *>(s)) {
    visitReturn(rs);
  }
}

void IRBuilder::visitBlock(BlockStmtNode *b) {
  pushScope();
  for (auto &s : b->body) visitStmt(s.get());
  popScope();
}

void IRBuilder::visitDeclStmt(DeclStmtNode *s) {
  for (auto &def : s->decl->defs)
    emitLocalVar(def.get(), s->decl->base, s->decl->isConst);
}

void IRBuilder::visitIf(IfStmtNode *s) {
  if (curBB_->hasTerminator()) curBB_ = newBlock();
  bool hasElse = s->elseStmt != nullptr;
  Value *cv = condValue(s->cond.get()); // may leave curBB_ at a merge block

  if (auto *ci = dynamic_cast<ConstantInt *>(cv)) {
    if (ci->v) visitStmt(s->thenStmt.get());
    else if (hasElse) visitStmt(s->elseStmt.get());
    return; // taken branch falls through (or already terminated)
  }

  BasicBlock *thenB = newBlock();
  BasicBlock *elseB = hasElse ? newBlock() : nullptr;
  BasicBlock *join = newBlock();

  Instruction *cb = newInstr(Op::CondBr, Type::Void, s->line);
  cb->ops = {cv, thenB, hasElse ? elseB : join};

  curBB_ = thenB;
  visitStmt(s->thenStmt.get());
  if (!curBB_->hasTerminator()) emitTerm(Op::Br, {join}, s->line);

  if (hasElse) {
    curBB_ = elseB;
    visitStmt(s->elseStmt.get());
    if (!curBB_->hasTerminator()) emitTerm(Op::Br, {join}, s->line);
  }
  curBB_ = join;
}

void IRBuilder::visitWhile(WhileStmtNode *s) {
  if (curBB_->hasTerminator()) curBB_ = newBlock();

  // ---- unstructured form --------------------------------------------------
  // A body that can `return`, `break` or `continue` out of this loop cannot
  // be captured by scf.while's regions; keep the plain cf.br/cf.cond_br form
  // (the same CFG the back-end used all along).
  if (!whileIsClean(s)) {
    BasicBlock *condB = newBlock();
    BasicBlock *bodyB = newBlock();
    BasicBlock *endB = newBlock();
    emitTerm(Op::Br, {condB}, s->line);

    curBB_ = condB;
    loops_.push_back({condB, endB});
    emitCondBr(s->cond.get(), bodyB, endB);

    curBB_ = bodyB;
    visitStmt(s->body.get());
    if (!curBB_->hasTerminator()) emitTerm(Op::Br, {condB}, s->line);
    loops_.pop_back();

    curBB_ = endB;
    return;
  }

  // ---- affine layer (top of the mid-end): canonical counting loops ---------
  // A clean `while (iv < ub) { body; iv = iv + step }` over a local i32 slot
  // is kept as an affine.for region op so the affine layer can optimise it
  // (IV-based analysis, unrolling, vectorisation...).  The mid-end pass
  // lower-affine-to-scf (midend/pass/Passes.cpp) turns it back into the
  // generic scf.while form for the middle layer.  Everything that does not
  // match stays scf.while.
  if (tryEmitAffineFor(s)) return;

  // ---- structured scf.while ----------------------------------------------
  // Kept in MLIR's scf.while shape, which the printer shows and which later
  // mid-end passes can exploit:
  //
  //   <pre code>
  //   scf.while {                 // condRegion
  //     ^cond:  ...cond...        // re-entered every iteration
  //             scf.condition %c  // true -> body region, false -> exit block
  //   } do {
  //     ^body:  ...body...        // bodyRegion
  //             scf.yield         // back to ^cond
  //   }
  //   <exit block>
  BasicBlock *exitB = newBlock(); // block the loop continues into when done
  Instruction *op = newInstr(Op::ScfWhile, Type::Void, s->line);
  op->ops = {exitB};

  // condition region
  regStack_.push_back(&op->condRegion);
  BasicBlock *condEntry = newBlock();
  curBB_ = condEntry;
  Value *cond = condValue(s->cond.get()); // may open more cond-region blocks

  // body-region entry must exist before scf.condition's targets are fixed
  regStack_.pop_back();
  regStack_.push_back(&op->bodyRegion);
  BasicBlock *bodyEntry = newBlock();
  regStack_.pop_back();

  Instruction *sc = newInstr(Op::ScfCondition, Type::Void, s->line);
  sc->ops = {cond, bodyEntry, exitB};

  // body region
  regStack_.push_back(&op->bodyRegion);
  curBB_ = bodyEntry;
  visitStmt(s->body.get());
  if (!curBB_->hasTerminator()) newInstr(Op::ScfYield, Type::Void, s->line);
  regStack_.pop_back();

  curBB_ = exitB;
}

// A `while` is structure-eligible only when its body subtree contains no
// `return` and no break/continue that would target *this* loop.  Returns and
// breaks inside a nested while are reported separately: returns still poison
// the outer loop (they would have to leave every enclosing region), whereas a
// nested break/continue only makes that inner loop unstructured.
void IRBuilder::scanLoopTree(StmtNode *n, bool &hasRet, bool &hasOwnBC) {
  if (!n) return;
  if (auto *blk = dynamic_cast<BlockStmtNode *>(n)) {
    for (auto &s : blk->body) scanLoopTree(s.get(), hasRet, hasOwnBC);
  } else if (auto *ifs = dynamic_cast<IfStmtNode *>(n)) {
    scanLoopTree(ifs->thenStmt.get(), hasRet, hasOwnBC);
    scanLoopTree(ifs->elseStmt.get(), hasRet, hasOwnBC);
  } else if (auto *ws = dynamic_cast<WhileStmtNode *>(n)) {
    bool r = false, b = false; // b is intentionally dropped here
    scanLoopTree(ws->body.get(), r, b);
    hasRet |= r;
  } else if (dynamic_cast<BreakStmtNode *>(n) ||
             dynamic_cast<ContinueStmtNode *>(n)) {
    hasOwnBC = true;
  } else if (dynamic_cast<ReturnStmtNode *>(n)) {
    hasRet = true;
  }
}

bool IRBuilder::whileIsClean(WhileStmtNode *s) {
  bool r = false, bc = false;
  scanLoopTree(s->body.get(), r, bc);
  return !r && !bc;
}

// ---- canonical counting loops (affine layer) ------------------------------
// Helpers shared by the affine.for matcher.  A counting loop is only lifted to
// affine when every property can be proven syntactically; anything uncertain
// simply stays in scf.while / cf, which is always a safe fallback.

namespace {

// Does an expression subtree contain a function call?  Used to decide whether
// a global (or otherwise externally reachable) memory object can change value
// between iterations.
bool exprHasCall(const ExprNode *e) {
  if (!e) return false;
  if (auto *lv = dynamic_cast<const LValNode *>(e)) {
    for (auto &ix : lv->indices)
      if (exprHasCall(ix.get())) return true;
    return false;
  }
  if (auto *un = dynamic_cast<const UnaryExprNode *>(e))
    return exprHasCall(un->operand.get());
  if (auto *bn = dynamic_cast<const BinaryExprNode *>(e))
    return exprHasCall(bn->lhs.get()) || exprHasCall(bn->rhs.get());
  if (dynamic_cast<const CallExprNode *>(e)) return true;
  return false;
}

bool initHasCall(const InitNode *in) {
  if (!in) return false;
  if (in->kind == InitNode::EXPR) return exprHasCall(in->expr.get());
  for (auto &c : in->list)
    if (initHasCall(c.get())) return true;
  return false;
}

} // namespace

bool IRBuilder::stmtTreeHasCall(StmtNode *s) {
  if (!s) return false;
  if (auto *blk = dynamic_cast<BlockStmtNode *>(s)) {
    for (auto &c : blk->body)
      if (stmtTreeHasCall(c.get())) return true;
    return false;
  }
  if (auto *ds = dynamic_cast<DeclStmtNode *>(s)) {
    for (auto &d : ds->decl->defs)
      if (initHasCall(d->init.get())) return true;
    return false;
  }
  if (auto *as = dynamic_cast<AssignStmtNode *>(s))
    return exprHasCall(as->rhs.get());
  if (auto *es = dynamic_cast<ExprStmtNode *>(s))
    return exprHasCall(es->expr.get());
  if (auto *ifs = dynamic_cast<IfStmtNode *>(s))
    return exprHasCall(ifs->cond.get()) || stmtTreeHasCall(ifs->thenStmt.get()) ||
           stmtTreeHasCall(ifs->elseStmt.get());
  if (auto *ws = dynamic_cast<WhileStmtNode *>(s))
    return exprHasCall(ws->cond.get()) || stmtTreeHasCall(ws->body.get());
  if (auto *rs = dynamic_cast<ReturnStmtNode *>(s))
    return exprHasCall(rs->value.get());
  return false;
}

bool IRBuilder::stmtTreeWritesName(StmtNode *s, const std::string &name,
                                   const StmtNode *skip) {
  if (!s || s == skip) return false;
  if (auto *blk = dynamic_cast<BlockStmtNode *>(s)) {
    for (auto &c : blk->body)
      if (stmtTreeWritesName(c.get(), name, skip)) return true;
    return false;
  }
  if (auto *ds = dynamic_cast<DeclStmtNode *>(s)) {
    // A declaration shadowing `name` means all following (inner) writes
    // concern that new object, but a *write* to it would still change what an
    // unshadowed read at the same point saw; conservatively reject those.
    for (auto &d : ds->decl->defs)
      if (d->name == name) return true;
    return false;
  }
  if (auto *as = dynamic_cast<AssignStmtNode *>(s)) {
    if (auto *lv = dynamic_cast<LValNode *>(as->lhs.get()))
      if (lv->name == name) return true;
    return false;
  }
  if (auto *ifs = dynamic_cast<IfStmtNode *>(s)) {
    if (stmtTreeWritesName(ifs->thenStmt.get(), name, skip)) return true;
    return stmtTreeWritesName(ifs->elseStmt.get(), name, skip);
  }
  if (auto *ws = dynamic_cast<WhileStmtNode *>(s))
    return stmtTreeWritesName(ws->body.get(), name, skip);
  return false;
}

bool IRBuilder::boundIsInvariant(const ExprNode *bound, StmtNode *body,
                                 const std::string &ivName) {
  // A compile-time constant can never change between iterations.
  if (bound->cval.valid) return true;
  if (bound->isTensorVal || bound->isArrayExpr) return false;
  if (bound->evalType != TypeCat::INT) return false;

  // The bound must be a plain scalar object read (no subscripts / tensors),
  // distinct from the induction variable.
  auto *lv = dynamic_cast<const LValNode *>(bound);
  if (!lv || !lv->indices.empty()) return false;
  if (lv->name == ivName) return false;

  Obj *o = lookup(lv->name);
  if (!o || o->isArray || !o->dims.empty()) return false;
  if (o->elem != Type::I32) return false;

  // The loop body must never assign the bound: the source re-reads it on
  // every iteration, the affine form evaluates it exactly once up front.
  if (stmtTreeWritesName(body, lv->name)) return false;

  // Local slots and parameters are private to this frame, so no call can
  // touch them; a global scalar is only safe when the body performs no calls.
  if (dynamic_cast<GlobalVar *>(o->addr)) return !stmtTreeHasCall(body);
  return true;
}

bool IRBuilder::matchCountingWhile(WhileStmtNode *s, CountLoop &out) {
  // Only *clean* loops can be lifted into a structured form at all (no return
  // and no break/continue that targets this loop).
  if (!whileIsClean(s)) return false;

  // The condition must be exactly `iv < ub` (or its mirror `ub > iv`), an
  // int comparison.  Anything else - `<=`, `!=`, compound conditions - is not
  // a canonical counting loop and stays in the scf layer.
  auto *cond = dynamic_cast<BinaryExprNode *>(s->cond.get());
  if (!cond || cond->evalType != TypeCat::INT) return false;
  LValNode *ivLv = nullptr;
  ExprNode *bound = nullptr;
  if (cond->op == BinaryOp::LT) {
    if (auto *lv = dynamic_cast<LValNode *>(cond->lhs.get())) {
      ivLv = lv;
      bound = cond->rhs.get();
    }
  } else if (cond->op == BinaryOp::GT) {
    if (auto *lv = dynamic_cast<LValNode *>(cond->rhs.get())) {
      ivLv = lv;
      bound = cond->lhs.get();
    }
  }
  if (!ivLv || !bound) return false;
  if (ivLv->isTensorVal || !ivLv->indices.empty()) return false;
  if (ivLv->evalType != TypeCat::INT) return false;

  // The induction variable must be a memory-resident local i32 scalar (the
  // lowering re-materialises the loop over its slot; globals could be written
  // by calls and are therefore refused).
  Obj *o = lookup(ivLv->name);
  if (!o || o->isArray || o->isConst || !o->dims.empty()) return false;
  auto *slot = dynamic_cast<Instruction *>(o->addr);
  if (!slot || slot->op != Op::Alloca || slot->elem != Type::I32) return false;

  // The tail of the loop body must be exactly `iv = iv + K` (K a positive int
  // constant).  The affine lowering turns it into the loop's step.
  StmtNode *body = s->body.get();
  StmtNode *tail = body;
  if (auto *blk = dynamic_cast<BlockStmtNode *>(body)) {
    if (blk->body.empty()) return false;
    tail = blk->body.back().get();
  }
  auto *as = dynamic_cast<AssignStmtNode *>(tail);
  if (!as) return false;
  auto *lv = dynamic_cast<LValNode *>(as->lhs.get());
  if (!lv || lv->name != ivLv->name || !lv->indices.empty() ||
      lv->isTensorVal)
    return false;
  auto *incArith = dynamic_cast<BinaryExprNode *>(as->rhs.get());
  if (!incArith || incArith->op != BinaryOp::ADD) return false;
  ExprNode *constSide = nullptr;
  if (auto *al = dynamic_cast<LValNode *>(incArith->lhs.get()))
    if (al->name == ivLv->name && al->indices.empty())
      constSide = incArith->rhs.get();
  if (!constSide)
    if (auto *ar = dynamic_cast<LValNode *>(incArith->rhs.get()))
      if (ar->name == ivLv->name && ar->indices.empty())
        constSide = incArith->lhs.get();
  if (!constSide || !constSide->cval.valid ||
      constSide->evalType != TypeCat::INT)
    return false;
  int64_t step = constSide->cval.ival;
  if (step <= 0 || step > (int64_t)INT32_MAX) return false;

  // No other statement in the body may write `iv`: the affine step runs at the
  // end of *every* iteration and would clobber any other value.
  if (stmtTreeWritesName(body, ivLv->name, tail)) return false;

  // The upper bound must not change during the loop.
  if (!boundIsInvariant(bound, body, ivLv->name)) return false;

  out.iv = ivLv->name;
  out.obj = o;
  out.bound = bound;
  out.step = step;
  return true;
}

void IRBuilder::emitBodyWithoutTail(StmtNode *body) {
  auto *blk = dynamic_cast<BlockStmtNode *>(body);
  if (!blk) return; // the body itself is the increment: nothing to emit
  pushScope();
  for (size_t i = 0; i + 1 < blk->body.size(); ++i)
    visitStmt(blk->body[i].get());
  popScope();
}

bool IRBuilder::tryEmitAffineFor(WhileStmtNode *s) {
  CountLoop cl;
  if (!matchCountingWhile(s, cl)) return false;
  if (curBB_->hasTerminator()) curBB_ = newBlock();

  Obj *o = cl.obj;
  Value *slot = o->addr;
  int line = s->line;

  // Evaluate the iteration-invariant upper bound exactly once, up front; the
  // lower bound is whatever the induction slot holds when the loop is entered
  // (the enclosing code stored the initial value there, if any).
  Value *ub = visitExpr(cl.bound);
  Value *lb = emitLoad(slot, Type::I32, line);

  BasicBlock *exitB = newBlock(); // where control continues when the loop ends
  Instruction *op = newInstr(Op::AffineFor, Type::Void, line);
  op->ops = {exitB, slot, lb, ub};
  op->cond = Cond::Lt; // canonical ascending count: iv < ub
  op->step = cl.step;

  // body region: all loop statements except the trailing increment, which is
  // re-materialised by the lower-affine-to-scf pass as `iv += step`.
  regStack_.push_back(&op->bodyRegion);
  BasicBlock *bodyEntry = newBlock();
  curBB_ = bodyEntry;
  emitBodyWithoutTail(s->body.get());
  if (!curBB_->hasTerminator()) newInstr(Op::AffineYield, Type::Void, line);
  regStack_.pop_back();

  curBB_ = exitB;
  return true;
}

void IRBuilder::visitReturn(ReturnStmtNode *s) {
  if (retIsTensorFn_) {
    // Write the whole tensor value into the hidden result buffer and return
    // nothing (SysY-level "return tensor" is lowered via sret).
    if (!s->value)
      throw CompileError("internal: tensor function missing return value",
                         s->line);
    emitTensorExprTo(s->value.get(), retBuf_);
    emitTerm(Op::Ret, {}, s->line);
    return;
  }
  if (!s->value) {
    emitTerm(Op::Ret, {}, s->line);
    return;
  }
  Value *v = visitExpr(s->value.get());
  v = convert(v, curFn_->ret, s->line);
  emitTerm(Op::Ret, {v}, s->line);
}

void IRBuilder::visitAssign(AssignStmtNode *s) {
  if (curBB_->hasTerminator()) curBB_ = newBlock();
  auto *lv = dynamic_cast<LValNode *>(s->lhs.get());
  if (!lv) throw CompileError("internal: assignment to non-lvalue", s->line);
  if (lv->isTensorVal) {
    // whole-tensor assignment: compute the RHS directly into the l-value's
    // storage buffer (elements are written in-place by tensor ops).
    emitTensorExprTo(s->rhs.get(), addressOf(lv));
    return;
  }
  Value *addr = addressOf(lv);
  Value *v = visitExpr(s->rhs.get());
  v = convert(v, typeOf(lv->evalType), s->line);
  emitStore(addr, v, s->line);
}

void IRBuilder::visitExprStmt(ExprStmtNode *s) {
  if (!s->expr) return;
  (void)visitExpr(s->expr.get());
}

// ---------------------------------------------------------------------------
// local variable declarations
// ---------------------------------------------------------------------------

void IRBuilder::emitLocalVar(VarDefNode *def, TypeCat base, bool isConst) {
  const VarType &t = def->type;
  Type elem = typeOf(base);

  if (t.dims.empty()) {
    if (isConst) return; // reads are folded to constants by semantic analysis
    Value *slot = emitAlloca(elem, 1, def->line);
    auto *o = new Obj;
    o->addr = slot;
    o->elem = elem;
    o->isConst = false;
    bind(def->name, o);
    if (def->init) {
      if (def->init->kind != InitNode::EXPR)
        throw CompileError("internal: scalar initializer", def->line);
      Value *v = visitExpr(def->init->expr.get());
      v = convert(v, elem, def->line);
      emitStore(slot, v, def->line);
    }
    return;
  }

  // array (const or not): allocate stack storage and materialise initializer
  int64_t total = t.numElements();
  Value *slot = emitAlloca(elem, total, def->line);
  auto *o = new Obj;
  o->addr = slot;
  o->elem = elem;
  o->isArray = true;
  o->isConst = isConst;
  o->dims = t.dims;
  bind(def->name, o);
  if (def->init) emitLocalArrayInit(t.dims, base, def->init.get(), slot);
}

namespace {

// Walk an initializer tree in row-major fill order and record for every scalar
// element either its source expression or "zero fill". Mirrors the const
// flattener used by the semantic analyser, but keeps the leaf expressions so
// that dynamic values can be evaluated at runtime.
class InitWalker {
public:
  struct Leaf {
    int64_t pos;
    const ExprNode *expr; // null => zero
  };

  InitWalker(const vector<int64_t> &dims, TypeCat base, int line)
      : dims_(dims), base_(base), line_(line) {}

  void run(InitNode *init) {
    if (init && init->kind == InitNode::EXPR) {
      if (dims_.size() == 1) { // allowed as `T a[1] = v;`
        leaves_.push_back({0, init->expr.get()});
        return;
      }
      throw CompileError("internal: scalar expression initializes an array",
                         line_);
    }
    if (!init) return; // nothing to store; stack garbage is allowed
    Stream s(init->list);
    fillAggregate(0, s);
    if (!s.empty())
      throw CompileError("internal: excess elements in array initializer",
                         line_);
  }

  const vector<Leaf> &leaves() const { return leaves_; }

private:
  const vector<int64_t> &dims_;
  TypeCat base_;
  int line_;
  vector<Leaf> leaves_;
  int64_t pos_ = 0;

  struct Stream {
    const vector<unique_ptr<InitNode>> &items;
    size_t idx = 0;
    explicit Stream(const vector<unique_ptr<InitNode>> &v) : items(v) {}
    bool empty() const { return idx >= items.size(); }
    InitNode *peek() const { return empty() ? nullptr : items[idx].get(); }
  };

  void pushScalarFrom(InitNode *item) {
    while (item && item->kind == InitNode::LIST) {
      if (item->list.size() != 1)
        throw CompileError("internal: invalid scalar initializer", line_);
      item = item->list[0].get();
    }
    if (!item || item->kind != InitNode::EXPR)
      throw CompileError("internal: invalid initializer", line_);
    leaves_.push_back({pos_++, item->expr.get()});
  }

  void consumeScalar(Stream &s) {
    if (s.empty()) {
      leaves_.push_back({pos_++, nullptr});
      return;
    }
    InitNode *item = s.peek();
    if (item->kind == InitNode::LIST) {
      const auto &sub = item->list;
      if (sub.empty())
        throw CompileError("internal: empty braces in initializer", line_);
      if (sub.size() != 1)
        throw CompileError("internal: excess elements in array initializer",
                           line_);
      s.idx++;
      Stream one(sub);
      pushScalarFrom(one.peek());
      return;
    }
    pushScalarFrom(item);
    s.idx++;
  }

  void fillAggregate(size_t level, Stream &s) {
    if (level == dims_.size()) {
      consumeScalar(s);
      return;
    }
    for (int64_t i = 0; i < dims_[level]; i++) {
      InitNode *head = s.peek();
      if (head && head->kind == InitNode::LIST) {
        const auto &sub = head->list;
        Stream subS(sub);
        if (level + 1 == dims_.size()) {
          s.idx++;
          if (subS.empty())
            throw CompileError("internal: empty braces in initializer", line_);
          pushScalarFrom(subS.peek());
          subS.idx++;
          if (!subS.empty())
            throw CompileError("internal: excess elements in array initializer",
                               line_);
        } else {
          s.idx++;
          fillAggregate(level + 1, subS);
          if (!subS.empty())
            throw CompileError("internal: excess elements in array initializer",
                               line_);
        }
      } else {
        if (level + 1 == dims_.size()) {
          consumeScalar(s);
        } else {
          Stream parent(s.items);
          parent.idx = s.idx;
          fillAggregate(level + 1, parent);
          s.idx = parent.idx;
        }
      }
    }
  }
};

} // namespace

void IRBuilder::emitLocalArrayInit(const vector<int64_t> &dims, TypeCat base,
                                   InitNode *init, Value *baseAddr) {
  Type elem = typeOf(base);
  InitWalker w(dims, base, 0);
  w.run(init);
  const auto &leaves = w.leaves();

  // A huge initialiser would otherwise unroll into one store per element,
  // which is quadratic to compile.  Lower it as a runtime zero-fill loop plus
  // individual stores for the (usually few) non-zero leaves.
  const int64_t kUnrollLimit = 4096;
  if (leaves.size() > kUnrollLimit) {
    emitArrayZeroFill(baseAddr, (int32_t)leaves.size(), elem, 0);
    for (const auto &leaf : leaves) {
      if (!leaf.expr) continue;                       // covered by the loop
      if (leaf.expr->cval.valid) {                    // constant: only zero
        bool z = base == TypeCat::FLOAT
                     ? (leaf.expr->cval.fval == 0.0f)
                     : (leaf.expr->cval.ival == 0);
        if (z) continue;
      }
      Value *v = visitExpr(const_cast<ExprNode *>(leaf.expr));
      v = convert(v, elem, 0);
      Value *addr = emitGep(baseAddr, cI((int32_t)(leaf.pos * 4)), 0);
      emitStore(addr, v, 0);
    }
    return;
  }

  for (const auto &leaf : leaves) {
    Value *v;
    if (leaf.expr) v = visitExpr(const_cast<ExprNode *>(leaf.expr));
    else v = base == TypeCat::FLOAT ? (Value *)cF(0.0f) : (Value *)cI(0);
    v = convert(v, elem, 0);
    Value *addr = emitGep(baseAddr, cI((int32_t)(leaf.pos * 4)), 0);
    emitStore(addr, v, 0);
  }
}

// Zero-fill `count` scalar elements starting at baseAddr with a tight IR loop
// (index lives in a stack slot so no phi is required).
void IRBuilder::emitArrayZeroFill(Value *baseAddr, int32_t count, Type elem,
                                  int line) {
  if (curBB_->hasTerminator()) curBB_ = newBlock();
  BasicBlock *condB = newBlock();
  BasicBlock *bodyB = newBlock();
  BasicBlock *endB = newBlock();

  Value *idxSlot = emitAlloca(Type::I32, 1, line);
  emitStore(idxSlot, cI(0), line);
  emitTerm(Op::Br, {condB}, line);

  curBB_ = condB;
  Value *i = emitLoad(idxSlot, Type::I32, line);
  Value *cmp = emitCmp(Cond::Lt, i, cI(count), false, line);
  Instruction *cb = newInstr(Op::CondBr, Type::Void, line);
  cb->ops = {cmp, bodyB, endB};

  curBB_ = bodyB;
  Value *ib = emitLoad(idxSlot, Type::I32, line);
  Instruction *mul = newInstr(Op::Mul, Type::I32, line);
  mul->ops = {ib, cI(4)};
  Value *addr = emitGep(baseAddr, mul, line);
  Value *zero =
      elem == Type::F32 ? (Value *)cF(0.0f) : (Value *)cI(0);
  emitStore(addr, zero, line);
  Instruction *inc = newInstr(Op::Add, Type::I32, line);
  inc->ops = {ib, cI(1)};
  emitStore(idxSlot, inc, line);
  emitTerm(Op::Br, {condB}, line);

  curBB_ = endB;
}

// ---------------------------------------------------------------------------
// expressions
// ---------------------------------------------------------------------------

Value *IRBuilder::visitExpr(ExprNode *e) {
  if (e->cval.valid) return constOf(e->cval);
  if (e->isTensorVal)
    throw CompileError("internal: whole-tensor value used as a scalar",
                       e->line);

  if (auto *lv = dynamic_cast<LValNode *>(e)) return visitLValValue(lv);

  if (auto *un = dynamic_cast<UnaryExprNode *>(e)) {
    switch (un->op) {
    case UnaryOp::PLUS:
      return visitExpr(un->operand.get());
    case UnaryOp::MINUS: {
      Value *v = visitExpr(un->operand.get());
      if (v->ty == Type::F32) {
        Instruction *i = newInstr(Op::FSub, Type::F32, e->line);
        i->ops = {cF(0.0f), v};
        return i;
      }
      Instruction *i = newInstr(Op::Sub, Type::I32, e->line);
      i->ops = {cI(0), v};
      return i;
    }
    case UnaryOp::NOT: {
      Value *b = condValue(un->operand.get());
      Instruction *i = newInstr(Op::Not, Type::I32, e->line);
      i->ops = {b};
      return i;
    }
    }
    throw CompileError("internal: unknown unary operator", e->line);
  }

  if (auto *bn = dynamic_cast<BinaryExprNode *>(e)) {
    BinaryOp op = bn->op;
    if (op == BinaryOp::AND || op == BinaryOp::OR) return evalLogical(bn);

    Value *l = visitExpr(bn->lhs.get());
    Value *r = visitExpr(bn->rhs.get());

    switch (op) {
    case BinaryOp::ADD:
    case BinaryOp::SUB:
    case BinaryOp::MUL:
    case BinaryOp::DIV: {
      Type rt = typeOf(bn->evalType);
      l = convert(l, rt, e->line);
      r = convert(r, rt, e->line);
      Op irOp;
      switch (op) {
      case BinaryOp::ADD: irOp = (rt == Type::F32) ? Op::FAdd : Op::Add; break;
      case BinaryOp::SUB: irOp = (rt == Type::F32) ? Op::FSub : Op::Sub; break;
      case BinaryOp::MUL: irOp = (rt == Type::F32) ? Op::FMul : Op::Mul; break;
      default: irOp = (rt == Type::F32) ? Op::FDiv : Op::SDiv; break;
      }
      Instruction *i = newInstr(irOp, rt, e->line);
      i->ops = {l, r};
      return i;
    }
    case BinaryOp::MOD: {
      Type rt = Type::I32;
      l = convert(l, rt, e->line);
      r = convert(r, rt, e->line);
      Instruction *i = newInstr(Op::SRem, Type::I32, e->line);
      i->ops = {l, r};
      return i;
    }
    case BinaryOp::LT:
    case BinaryOp::GT:
    case BinaryOp::LE:
    case BinaryOp::GE:
    case BinaryOp::EQ:
    case BinaryOp::NE: {
      bool isF = (l->ty == Type::F32 || r->ty == Type::F32);
      if (isF) {
        l = convert(l, Type::F32, e->line);
        r = convert(r, Type::F32, e->line);
      }
      Cond c;
      switch (op) {
      case BinaryOp::LT: c = Cond::Lt; break;
      case BinaryOp::GT: c = Cond::Gt; break;
      case BinaryOp::LE: c = Cond::Le; break;
      case BinaryOp::GE: c = Cond::Ge; break;
      case BinaryOp::EQ: c = Cond::Eq; break;
      default: c = Cond::Ne; break;
      }
      return emitCmp(c, l, r, isF, e->line);
    }
    default:
      throw CompileError("internal: unexpected binary operator", e->line);
    }
  }

  if (auto *cl = dynamic_cast<CallExprNode *>(e)) {
    return lowerCallCodegen(cl, nullptr);
  }

  throw CompileError("internal: unknown expression node", e->line);
}

Value *IRBuilder::visitLValValue(LValNode *lv) {
  if (lv->isTensorVal)
    throw CompileError("internal: whole-tensor lvalue used as a scalar",
                       lv->line);
  Value *addr = addressOf(lv);
  return emitLoad(addr, typeOf(lv->evalType), lv->line);
}

Value *IRBuilder::addressOf(LValNode *lv) {
  Obj *o = lookup(lv->name);
  if (!o) {
    // A scalar constant never needs storage; only reachable for malformed IR.
    throw CompileError("internal: unresolved variable '" + lv->name + "'",
                       lv->line);
  }
  const vector<int64_t> &dims = o->dims;
  if (dims.empty()) {
    if (!lv->indices.empty())
      throw CompileError("internal: subscript on scalar", lv->line);
    return o->addr;
  }
  if (lv->indices.empty()) return o->addr;

  // Row-major byte offset from the leading subscripts:
  //   off = sum_p i_p * (prod_{q>p} dims[q]) * elemBytes
  // Constant terms are folded (mod 2^32, matching 32-bit runtime arithmetic);
  // dynamic terms become an i32 multiply-add chain.
  size_t m = dims.size();
  size_t k = lv->indices.size();
  if (k > m)
    throw CompileError("internal: too many subscripts", lv->line);

  int64_t constOff = 0;
  Value *dyn = nullptr;
  for (size_t p = 0; p < k; ++p) {
    Value *idx = visitExpr(lv->indices[p].get()); // i32
    int64_t scale = 1;
    for (size_t q = p + 1; q < m; ++q) {
      if (dims[q] > 0) scale *= dims[q];
    }
    scale *= 4; // element byte size
    if (auto *ci = dynamic_cast<ConstantInt *>(idx)) {
      constOff += (int64_t)ci->v * scale;
      continue;
    }
    Value *term = idx;
    if (scale != 1) {
      Instruction *mul = newInstr(Op::Mul, Type::I32, lv->line);
      mul->ops = {term, cI((int32_t)scale)};
      term = mul;
    }
    if (dyn) {
      Instruction *add = newInstr(Op::Add, Type::I32, lv->line);
      add->ops = {dyn, term};
      dyn = add;
    } else {
      dyn = term;
    }
  }
  if (constOff != 0) {
    if (dyn) {
      Instruction *add = newInstr(Op::Add, Type::I32, lv->line);
      add->ops = {dyn, cI((int32_t)constOff)};
      dyn = add;
    } else {
      dyn = cI((int32_t)constOff);
    }
  }
  if (!dyn) return o->addr;
  return emitGep(o->addr, dyn, lv->line);
}

Value *IRBuilder::codegenArg(ExprNode *e, const ParamDesc &formal, int line) {
  if (formal.isArray) {
    auto *lv = dynamic_cast<LValNode *>(e);
    if (!lv)
      throw CompileError("internal: array argument is not an lvalue", line);
    return addressOf(lv);
  }
  return convert(visitExpr(e), formal.ty, line);
}

// ---------------------------------------------------------------------------
// whole-tensor lowering (TensorType)
// ---------------------------------------------------------------------------

Instruction *IRBuilder::lowerCallCodegen(CallExprNode *cl, Value *sretDest) {
  Function *fn = findOrDeclareFunc(cl->name);
  auto fit = funcs_->find(cl->name);
  if (fit == funcs_->end())
    throw CompileError("internal: call to undeclared function '" + cl->name +
                           "'",
                       cl->line);
  const FuncInfo &fi = fit->second;

  vector<Value *> args;
  if (sretDest) args.push_back(sretDest); // tensor-returning: hidden buffer
  size_t off = fi.retIsTensor ? 1 : 0;
  for (size_t k = 0; k < cl->args.size(); ++k) {
    if (k + off >= fn->params.size())
      throw CompileError("internal: argument count mismatch", cl->line);
    const ParamDesc &fd = fn->params[k + off];
    ExprNode *ae = cl->args[k].get();
    if (fi.params[k].isTensor) {
      // whole-tensor argument: pass its buffer (evaluating computed values
      // into a fresh temporary first)
      Value *buf;
      if (auto *lv = dynamic_cast<LValNode *>(ae)) {
        buf = addressOf(lv);
      } else {
        buf = bufferOfTensor(ae);
      }
      args.push_back(buf);
    } else {
      args.push_back(codegenArg(ae, fd, cl->line));
    }
  }
  Instruction *i = newInstr(Op::Call, sretDest ? Type::Void : fn->ret,
                            cl->line);
  i->ops.push_back(fn);
  for (Value *a : args) i->ops.push_back(a);
  return i;
}

void IRBuilder::emitTensorCopy(Value *dest, Value *src, Type elem,
                               const std::vector<int64_t> &shape, int line) {
  Instruction *i = newInstr(Op::TCopy, Type::Void, line);
  i->ops = {dest, src};
  i->elem = elem;
  i->shape = shape;
}

Instruction *IRBuilder::tensorOp(Op op, Type elem,
                                 const std::vector<int64_t> &shape,
                                 Value *dest, const std::vector<Value *> &ops,
                                 int line) {
  Instruction *i = newInstr(op, Type::Void, line);
  i->elem = elem;
  i->shape = shape;
  i->ops.push_back(dest);
  for (Value *o : ops) i->ops.push_back(o);
  return i;
}

Value *IRBuilder::bufferOfTensor(ExprNode *e) {
  if (auto *lv = dynamic_cast<LValNode *>(e)) return addressOf(lv);
  // computed tensor value: materialise it into a fresh destination buffer
  Type elem = typeOf(e->evalType);
  int64_t total = 1;
  for (int64_t d : e->tShape) total *= d;
  Value *tmp = emitAlloca(elem, total, e->line);
  emitTensorExprTo(e, tmp);
  return tmp;
}

void IRBuilder::emitTensorExprTo(ExprNode *e, Value *dest) {
  int line = e->line;
  if (auto *lv = dynamic_cast<LValNode *>(e)) {
    // reading a whole tensor object: copy its storage into dest (unless it
    // already is dest)
    Value *src = addressOf(lv);
    if (src != dest) {
      emitTensorCopy(dest, src, typeOf(lv->evalType), lv->varDims, line);
    }
    return;
  }
  if (auto *un = dynamic_cast<UnaryExprNode *>(e)) {
    lowerTensorUnary(un, dest);
    return;
  }
  if (auto *bn = dynamic_cast<BinaryExprNode *>(e)) {
    lowerTensorBinary(bn, dest);
    return;
  }
  if (auto *cl = dynamic_cast<CallExprNode *>(e)) {
    lowerTensorCall(cl, dest);
    return;
  }
  throw CompileError("internal: whole-tensor expression node", line);
}

void IRBuilder::lowerTensorUnary(UnaryExprNode *un, Value *dest) {
  ExprNode *op = un->operand.get();
  if (un->op == UnaryOp::PLUS) {
    emitTensorExprTo(op, dest); // unary plus is the identity
    return;
  }
  // MINUS: elementwise negation into dest
  Value *src = bufferOfTensor(op);
  Type elem = typeOf(op->evalType);
  tensorOp(elem == Type::F32 ? Op::TNegF : Op::TNegI, elem, op->tShape, dest,
           {src}, un->line);
}

void IRBuilder::lowerTensorBinary(BinaryExprNode *bn, Value *dest) {
  int line = bn->line;
  Type elem = typeOf(bn->evalType);
  ExprNode *l = bn->lhs.get(), *r = bn->rhs.get();

  if (bn->op == BinaryOp::MATMUL) {
    Value *a = bufferOfTensor(l);
    Value *b = bufferOfTensor(r);
    // matrix multiply reads every source element many times, so a source that
    // aliases the destination must first be copied away
    auto copyAway = [&](Value *src, const std::vector<int64_t> &shape) {
      if (src != dest) return src;
      int64_t total = 1;
      for (int64_t d : shape) total *= d;
      Value *t = emitAlloca(elem, total, line);
      emitTensorCopy(t, src, elem, shape, line);
      return t;
    };
    a = copyAway(a, l->tShape);
    b = copyAway(b, r->tShape);
    std::vector<int64_t> mnp = {bn->tShape[0], l->tShape[1], bn->tShape[1]};
    tensorOp(elem == Type::F32 ? Op::TMatmulF : Op::TMatmulI, elem, mnp, dest,
             {a, b}, line);
    return;
  }

  // element-wise op with scalar promotion (a scalar operand is broadcast)
  auto operandOf = [&](ExprNode *sub) -> Value * {
    if (sub->isTensorVal) return bufferOfTensor(sub);
    return convert(visitExpr(sub), elem, line);
  };
  Value *a = operandOf(l);
  Value *b = operandOf(r);

  Op op;
  if (elem == Type::F32) {
    switch (bn->op) {
    case BinaryOp::ADD: op = Op::TAddF; break;
    case BinaryOp::SUB: op = Op::TSubF; break;
    case BinaryOp::MUL: op = Op::TMulF; break;
    case BinaryOp::DIV: op = Op::TDivF; break;
    default: throw CompileError("internal: invalid float tensor op", line);
    }
  } else {
    switch (bn->op) {
    case BinaryOp::ADD: op = Op::TAddI; break;
    case BinaryOp::SUB: op = Op::TSubI; break;
    case BinaryOp::MUL: op = Op::TMulI; break;
    case BinaryOp::DIV: op = Op::TDivI; break;
    case BinaryOp::MOD: op = Op::TRemI; break;
    default: throw CompileError("internal: invalid int tensor op", line);
    }
  }
  tensorOp(op, elem, bn->tShape, dest, {a, b}, line);
}

void IRBuilder::lowerTensorCall(CallExprNode *cl, Value *dest) {
  lowerCallCodegen(cl, dest); // callee writes its result into `dest` (sret)
}

// ---------------------------------------------------------------------------
// conditions / short-circuit
// ---------------------------------------------------------------------------

Value *IRBuilder::condValue(ExprNode *e) {
  if (e->cval.valid) {
    const ConstVal &c = e->cval;
    bool t = (c.type == TypeCat::FLOAT) ? (c.fval != 0.0f) : (c.ival != 0);
    return cI(t ? 1 : 0);
  }
  if (auto *bn = dynamic_cast<BinaryExprNode *>(e)) {
    if (bn->op == BinaryOp::AND || bn->op == BinaryOp::OR)
      return evalLogical(bn);
  }
  // Comparisons and logical negation already yield 0/1.
  bool alreadyBool = false;
  if (auto *b2 = dynamic_cast<BinaryExprNode *>(e)) {
    switch (b2->op) {
    case BinaryOp::LT: case BinaryOp::GT: case BinaryOp::LE:
    case BinaryOp::GE: case BinaryOp::EQ: case BinaryOp::NE:
      alreadyBool = true;
      break;
    default: break;
    }
  }
  if (auto *un = dynamic_cast<UnaryExprNode *>(e))
    if (un->op == UnaryOp::NOT) alreadyBool = true;
  if (alreadyBool) return visitExpr(e);

  Value *v = visitExpr(e);
  if (v->ty == Type::I32) return emitCmp(Cond::Ne, v, cI(0), false, e->line);
  return emitCmp(Cond::Ne, v, cF(0.0f), true, e->line);
}

Value *IRBuilder::evalLogical(BinaryExprNode *e) {
  bool isAnd = (e->op == BinaryOp::AND);
  if (curBB_->hasTerminator()) curBB_ = newBlock();

  Value *l = condValue(e->lhs.get());

  // Short-cut when the left side decides the result: the right side must not
  // be evaluated (and need not even be code-generated).
  if (auto *li = dynamic_cast<ConstantInt *>(l)) {
    if (isAnd) return li->v == 0 ? (Value *)cI(0) : condValue(e->rhs.get());
    return li->v != 0 ? (Value *)cI(1) : condValue(e->rhs.get());
  }

  Value *slot = emitAlloca(Type::I32, 1, e->line);
  BasicBlock *altB = newBlock();   // AND: lhs false / OR: lhs true
  BasicBlock *rhsB = newBlock();
  BasicBlock *join = newBlock();

  Instruction *cb = newInstr(Op::CondBr, Type::Void, e->line);
  cb->ops = {l, isAnd ? rhsB : altB, isAnd ? altB : rhsB};

  curBB_ = altB;
  emitStore(slot, isAnd ? (Value *)cI(0) : (Value *)cI(1), e->line);
  emitTerm(Op::Br, {join}, e->line);

  curBB_ = rhsB;
  Value *r = condValue(e->rhs.get());
  emitStore(slot, r, e->line);
  emitTerm(Op::Br, {join}, e->line);

  curBB_ = join;
  return emitLoad(slot, Type::I32, e->line);
}

void IRBuilder::emitCondBr(ExprNode *cond, BasicBlock *T, BasicBlock *F) {
  Value *c = condValue(cond);
  if (auto *ci = dynamic_cast<ConstantInt *>(c)) {
    emitTerm(Op::Br, {ci->v ? T : F}, cond->line);
    return;
  }
  Instruction *i = newInstr(Op::CondBr, Type::Void, cond->line);
  i->ops = {c, T, F};
}

} // namespace ir
} // namespace sakura
