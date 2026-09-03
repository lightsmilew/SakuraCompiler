// Module/instruction factories and the MLIR-style printer.
#include "IR.h"

#include <functional>
#include <sstream>

namespace sakura {
namespace ir {

bool BasicBlock::hasTerminator() const {
  if (instrs.empty()) return false;
  Op o = instrs.back()->op;
  return o == Op::Ret || o == Op::Br || o == Op::CondBr;
}

ConstantInt *Module::constInt(int32_t v) {
  auto *c = new ConstantInt(v);
  arena_.emplace_back(c);
  return c;
}

ConstantFloat *Module::constFloat(float v) {
  auto *c = new ConstantFloat(v);
  arena_.emplace_back(c);
  return c;
}

GlobalVar *Module::addGlobal(const std::string &name, Type elem, int64_t elems,
                             bool isConst, bool hasInit,
                             const std::vector<ConstVal> &init) {
  auto g = std::make_unique<GlobalVar>(name, elem, elems, isConst);
  g->hasInit = hasInit;
  g->init = init;
  GlobalVar *p = g.get();
  globals_.emplace_back(std::move(g));
  globalMap_[name] = p;
  return p;
}

Function *Module::addFunction(const std::string &name, Type ret,
                              const std::vector<ParamDesc> &params,
                              bool isLib) {
  auto f = std::make_unique<Function>();
  f->name = name;
  f->ty = Type::Void;
  f->ret = ret;
  f->params = params;
  f->isLib = isLib;
  if (!isLib) {
    for (size_t i = 0; i < params.size(); ++i) {
      auto a = std::make_unique<Argument>();
      a->index = (int)i;
      a->name = params[i].name;
      a->ty = params[i].ty;
      f->args.emplace_back(std::move(a));
    }
  }
  Function *p = f.get();
  functions_.emplace_back(std::move(f));
  funcMap_[name] = p;
  return p;
}

GlobalVar *Module::getGlobal(const std::string &name) const {
  auto it = globalMap_.find(name);
  return it == globalMap_.end() ? nullptr : it->second;
}

Function *Module::getFunction(const std::string &name) const {
  auto it = funcMap_.find(name);
  return it == funcMap_.end() ? nullptr : it->second;
}

void Module::reserveBuiltin(const std::string &name) {
  if (funcMap_.count(name)) return;
  // Builtins have known signatures; register them lazily as needed.
  (void)name;
}

// ---------------------------------------------------------------------------
// MLIR dialect metadata
// ---------------------------------------------------------------------------

const char *dialectName(Dialect d) {
  switch (d) {
  case Dialect::Func:   return "func";
  case Dialect::Arith:  return "arith";
  case Dialect::Cf:     return "cf";
  case Dialect::Memref: return "memref";
  default:              return "tensor";
  }
}

Dialect dialectOf(Op o) {
  switch (o) {
  case Op::Ret:
  case Op::Call:
    return Dialect::Func;
  case Op::Br:
  case Op::CondBr:
    return Dialect::Cf;
  case Op::Alloca:
  case Op::Load:
  case Op::Store:
  case Op::Gep:
    return Dialect::Memref;
  case Op::TAddI: case Op::TAddF: case Op::TSubI: case Op::TSubF:
  case Op::TMulI: case Op::TMulF: case Op::TDivI: case Op::TDivF:
  case Op::TRemI: case Op::TNegI: case Op::TNegF: case Op::TCopy:
  case Op::TMatmulI: case Op::TMatmulF:
    return Dialect::Tensor;
  default:
    return Dialect::Arith;
  }
}

const char *opMnemonic(Op o) {
  switch (o) {
  case Op::Ret:    return "return";
  case Op::Call:   return "call";
  case Op::Br:     return "br";
  case Op::CondBr: return "cond_br";
  case Op::Alloca: return "alloca";
  case Op::Load:   return "load";
  case Op::Store:  return "store";
  case Op::Gep:    return "gep";
  case Op::Add:    return "addi";
  case Op::Sub:    return "subi";
  case Op::Mul:    return "muli";
  case Op::SDiv:   return "divsi";
  case Op::SRem:   return "remsi";
  case Op::FAdd:   return "addf";
  case Op::FSub:   return "subf";
  case Op::FMul:   return "mulf";
  case Op::FDiv:   return "divf";
  case Op::ICmp:   return "cmpi";
  case Op::FCmp:   return "cmpf";
  case Op::Sitofp: return "sitofp";
  case Op::Fptosi: return "fptosi";
  case Op::Not:    return "xori";
  case Op::TAddI:  return "addi";
  case Op::TAddF:  return "addf";
  case Op::TSubI:  return "subi";
  case Op::TSubF:  return "subf";
  case Op::TMulI:  return "muli";
  case Op::TMulF:  return "mulf";
  case Op::TDivI:  return "divsi";
  case Op::TDivF:  return "divf";
  case Op::TRemI:  return "remsi";
  case Op::TNegI:  return "negi";
  case Op::TNegF:  return "negf";
  case Op::TCopy:  return "copy";
  case Op::TMatmulI: return "matmul";
  case Op::TMatmulF: return "matmul";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// MLIR-style printer
// ---------------------------------------------------------------------------

namespace {

// format a float with enough digits to round-trip an f32
std::string fmtFloat(float v) {
  std::ostringstream s;
  s.precision(9);
  s << v;
  return s.str();
}

std::string intInit(const ConstVal &c) { return std::to_string(c.ival); }

std::string floatInit(const ConstVal &c) { return fmtFloat(c.fval); }

// memory shape of an addressable object: "i32" or "[4 x i32]"
std::string memDesc(Type elem, int64_t elems) {
  std::string e = typeStr(elem);
  if (elems == 1) return e;
  return "[" + std::to_string(elems) + " x " + e + "]";
}

// tensor buffer type: "memref<4xi32>", "memref<2x3xf32>"
std::string memrefT(Type elem, const std::vector<int64_t> &dims) {
  std::string e = typeStr(elem);
  std::string d;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i) d += "x";
    d += std::to_string(dims[i]);
  }
  if (dims.empty()) return e;
  return "memref<" + d + "x" + e + ">";
}

// type descriptor of the k-th operand of a tensor instruction
std::string tensorOperandT(const Instruction *inst, size_t k,
                           const std::vector<int64_t> &dims) {
  const Value *v = inst->ops[k];
  if (v->ty == Type::Ptr) return memrefT(inst->elem, dims);
  return typeStr(v->ty); // broadcast scalar
}

} // namespace

std::string Module::toString() {
  std::ostringstream os;
  os << "// SakuraCompiler MLIR-style module '" << name_ << "'\n";
  os << "module {\n";

  // ---- global objects --------------------------------------------------
  for (const auto &g : globals_) {
    os << "  memref.global @" << g->name << " : " << memDesc(g->elem, g->elems);
    if (g->hasInit) {
      if (g->elems == 1) {
        os << " = "
           << (g->elem == Type::I32 ? intInit(g->init[0]) : floatInit(g->init[0]));
      } else {
        os << " = [";
        for (int64_t i = 0; i < g->elems; ++i) {
          if (i) os << ", ";
          os << typeStr(g->elem) << " "
             << (g->elem == Type::I32 ? intInit(g->init[(size_t)i])
                                      : floatInit(g->init[(size_t)i]));
        }
        os << "]";
      }
    }
    os << "\n";
  }

  // ---- library declarations -------------------------------------------
  for (const auto &f : functions_) {
    if (!f->isLib) continue;
    os << "  func.func private @" << f->name << "(";
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (i) os << ", ";
      os << (f->params[i].isArray ? "ptr" : typeStr(f->params[i].ty));
    }
    os << ")";
    if (f->ret != Type::Void) os << " -> " << typeStr(f->ret);
    os << "\n";
  }

  // ---- value-name helpers per function --------------------------------
  for (const auto &f : functions_) {
    if (f->isLib) continue;

    // SSA result names (%0, %1, ...) in block order
    std::unordered_map<const Instruction *, std::string> ssa;
    int seq = 0;
    for (const auto &b : f->blocks)
      for (const auto &inst : b->instrs)
        if (inst->hasResult()) ssa[inst.get()] = "%" + std::to_string(seq++);

    // block labels ^bb0, ^bb1, ...  (the entry label is only printed when a
    // branch targets it, matching MLIR's canonical output)
    std::unordered_map<const BasicBlock *, std::string> blk;
    for (size_t i = 0; i < f->blocks.size(); ++i)
      blk[f->blocks[i].get()] = "^bb" + std::to_string(i);
    bool entryTargeted = false;
    if (!f->blocks.empty()) {
      const BasicBlock *entry = f->blocks.front().get();
      for (const auto &b : f->blocks)
        for (const auto &inst : b->instrs)
          if (inst->op == Op::Br || inst->op == Op::CondBr)
            for (Value *v : inst->ops)
              if (v == entry) entryTargeted = true;
    }

    // hoisted constants (arith.constant in the entry block)
    struct CEntry {
      const ConstantInt *ci = nullptr;
      const ConstantFloat *cf = nullptr;
      bool one = false;              // virtual `1 : i32` for arith.xori
    };
    std::vector<CEntry> consts;
    auto addCI = [&](const ConstantInt *c) {
      for (auto &e : consts)
        if ((e.ci && e.ci->v == c->v) || (e.one && c->v == 1))
          return;
      consts.push_back({c, nullptr, false});
    };
    auto addCF = [&](const ConstantFloat *c) {
      for (auto &e : consts)
        if (e.cf && e.cf->bits() == c->bits()) return;
      consts.push_back({nullptr, c, false});
    };
    auto hasIntOne = [&]() {
      for (auto &e : consts)
        if ((e.ci && e.ci->v == 1) || e.one) return true;
      return false;
    };
    bool anyNot = false;
    for (const auto &b : f->blocks)
      for (const auto &inst : b->instrs) {
        if (inst->op == Op::Not) anyNot = true;
        for (Value *v : inst->ops) {
          if (auto *ci = dynamic_cast<ConstantInt *>(v)) addCI(ci);
          else if (auto *cf = dynamic_cast<ConstantFloat *>(v)) addCF(cf);
        }
      }
    if (anyNot && !hasIntOne()) consts.push_back({nullptr, nullptr, true});

    std::vector<std::string> constName(consts.size());
    for (size_t i = 0; i < consts.size(); ++i)
      constName[i] = "%c" + std::to_string(i);

    auto constNameOf = [&](Value *v) -> std::string {
      if (auto *ci = dynamic_cast<ConstantInt *>(v)) {
        for (size_t i = 0; i < consts.size(); ++i) {
          const CEntry &e = consts[i];
          if (e.ci == ci || (e.ci && e.ci->v == ci->v) ||
              (e.one && ci->v == 1))
            return constName[i];
        }
        return "%?";
      }
      if (auto *cf = dynamic_cast<ConstantFloat *>(v)) {
        for (size_t i = 0; i < consts.size(); ++i) {
          const CEntry &e = consts[i];
          if (e.cf && e.cf->bits() == cf->bits()) return constName[i];
        }
        return "%?";
      }
      return "";
    };
    // resolved SSA operand name for any value
    std::function<std::string(Value *)> nameOf = [&](Value *v) -> std::string {
      if (dynamic_cast<ConstantInt *>(v) || dynamic_cast<ConstantFloat *>(v))
        return constNameOf(v);
      if (auto *g = dynamic_cast<GlobalVar *>(v)) return "@" + g->name;
      if (auto *bb = dynamic_cast<BasicBlock *>(v)) return blk[bb];
      if (auto *fn = dynamic_cast<Function *>(v)) return "@" + fn->name;
      if (auto *a = dynamic_cast<Argument *>(v))
        return "%arg" + std::to_string(a->index);
      if (auto *inst = dynamic_cast<Instruction *>(v)) {
        auto it = ssa.find(inst);
        return it != ssa.end() ? it->second : "%?";
      }
      return "%?";
    };

    // comparison predicates are formatted inline below (per-op), so nothing
    // is kept here.

    // ---- function header ------------------------------------------------
    os << "\n  func.func @" << f->name << "(";
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (i) os << ", ";
      os << "%arg" << i << ": "
         << (f->params[i].isArray ? "ptr" : typeStr(f->params[i].ty));
    }
    os << ")";
    if (f->ret != Type::Void) os << " -> " << typeStr(f->ret);
    os << " {\n";

    // ---- body ------------------------------------------------------------
    bool firstBlock = true;
    for (size_t bi = 0; bi < f->blocks.size(); ++bi) {
      const auto &b = f->blocks[bi];
      bool isEntry = (bi == 0);
      if (!(isEntry && !entryTargeted)) os << "  " << blk[b.get()] << ":\n";
      if (firstBlock) {
        // hoist constants into the entry block so they dominate every use
        for (size_t i = 0; i < consts.size(); ++i) {
          os << "    " << constName[i] << " = arith.constant ";
          const CEntry &e = consts[i];
          if (e.one) os << "1 : i32";
          else if (e.ci) os << e.ci->v << " : i32";
          else {
            os << fmtFloat(e.cf->v) << " : f32 ; 0x" << std::hex
               << e.cf->bits() << std::dec;
          }
          os << "\n";
        }
        firstBlock = false;
      }
      for (const auto &inst : b->instrs) {
        std::string res;
        if (inst->hasResult()) res = ssa.at(inst.get()) + " = ";
        os << "    " << res;
        switch (inst->op) {
        case Op::Ret: {
          os << "func.return";
          if (!inst->ops.empty()) os << " " << nameOf(inst->ops[0]);
          break;
        }
        case Op::Br: {
          os << "cf.br " << nameOf(inst->ops[0]);
          break;
        }
        case Op::CondBr: {
          os << "cf.cond_br " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]) << ", " << nameOf(inst->ops[2]);
          break;
        }
        case Op::Alloca: {
          os << "memref.alloca : " << memDesc(inst->elem, inst->n);
          break;
        }
        case Op::Load: {
          os << "memref.load " << nameOf(inst->ops[0]) << " : "
             << typeStr(inst->ty);
          break;
        }
        case Op::Store: {
          os << "memref.store " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]) << " : " << typeStr(inst->ops[0]->ty);
          break;
        }
        case Op::Gep: {
          os << "memref.gep " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]);
          break;
        }
        case Op::Add: case Op::Sub: case Op::Mul: case Op::SDiv: case Op::SRem:
        case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv: {
          os << opName(inst->op) << " " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]) << " : " << typeStr(inst->ty);
          break;
        }
        case Op::ICmp: case Op::FCmp: {
          bool isFloat = inst->op == Op::FCmp;
          const char *pred = nullptr;
          switch (inst->cond) {
          case Cond::Eq: pred = isFloat ? "oeq" : "eq"; break;
          case Cond::Ne: pred = isFloat ? "une" : "ne"; break;
          case Cond::Lt: pred = isFloat ? "olt" : "slt"; break;
          case Cond::Le: pred = isFloat ? "ole" : "sle"; break;
          case Cond::Gt: pred = isFloat ? "ogt" : "sgt"; break;
          case Cond::Ge: pred = isFloat ? "oge" : "sge"; break;
          }
          os << opName(inst->op) << " " << pred << ", "
             << nameOf(inst->ops[0]) << ", " << nameOf(inst->ops[1]) << " : "
             << typeStr(inst->ops[0]->ty);
          break;
        }
        case Op::Sitofp: case Op::Fptosi: {
          os << opName(inst->op) << " " << nameOf(inst->ops[0]) << " : "
             << typeStr(inst->ops[0]->ty) << " to " << typeStr(inst->ty);
          break;
        }
        case Op::Not: {
          os << "arith.xori " << nameOf(inst->ops[0]);
          // second operand: `1 : i32` (synthesised in the constant pool)
          for (size_t i = 0; i < consts.size(); ++i) {
            const CEntry &e = consts[i];
            if ((e.ci && e.ci->v == 1) || e.one) {
              os << ", " << constName[i] << " : i32";
              break;
            }
          }
          break;
        }
        case Op::Call: {
          auto *callee = static_cast<Function *>(inst->ops[0]);
          os << "func.call @" << callee->name << "(";
          for (size_t i = 1; i < inst->ops.size(); ++i) {
            if (i > 1) os << ", ";
            os << nameOf(inst->ops[i]);
          }
          os << ") : (";
          for (size_t i = 1; i < inst->ops.size(); ++i) {
            if (i > 1) os << ", ";
            os << typeStr(inst->ops[i]->ty);
          }
          os << ") -> " << (callee->ret == Type::Void ? "()" : typeStr(callee->ret));
          break;
        }
        case Op::TAddI: case Op::TAddF: case Op::TSubI: case Op::TSubF:
        case Op::TMulI: case Op::TMulF: case Op::TDivI: case Op::TDivF:
        case Op::TRemI: {
          // elementwise tensor arithmetic: tensor.addf %dst, %a, %b
          // (scalar operands are broadcast; whole-tensor operands are buffers)
          os << opName(inst->op) << " " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]) << ", " << nameOf(inst->ops[2]) << " : ";
          os << "(" << tensorOperandT(inst.get(), 1, inst->shape) << ", "
             << tensorOperandT(inst.get(), 2, inst->shape) << ") -> "
             << memrefT(inst->elem, inst->shape);
          break;
        }
        case Op::TNegI: case Op::TNegF: {
          os << opName(inst->op) << " " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]) << " : "
             << tensorOperandT(inst.get(), 1, inst->shape) << " -> "
             << memrefT(inst->elem, inst->shape);
          break;
        }
        case Op::TCopy: {
          os << "tensor.copy " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]) << " : "
             << memrefT(inst->elem, inst->shape);
          break;
        }
        case Op::TMatmulI: case Op::TMatmulF: {
          // shape holds {M, N, P}: dest[M,P] = a[M,N] @ b[N,P]
          os << opName(inst->op) << " " << nameOf(inst->ops[0]) << ", "
             << nameOf(inst->ops[1]) << ", " << nameOf(inst->ops[2]) << " : (";
          std::vector<int64_t> mN = {inst->shape[0], inst->shape[1]};
          std::vector<int64_t> nP = {inst->shape[1], inst->shape[2]};
          std::vector<int64_t> mP = {inst->shape[0], inst->shape[2]};
          os << memrefT(inst->elem, mN) << ", " << memrefT(inst->elem, nP)
             << ") -> " << memrefT(inst->elem, mP);
          break;
        }
        }
        os << "\n";
      }
    }
    os << "  }\n";
  }

  os << "}\n";
  return os.str();
}

} // namespace ir
} // namespace sakura
