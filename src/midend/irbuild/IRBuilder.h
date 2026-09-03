// ---------------------------------------------------------------------------
// IRBuilder.h: lowers the *checked* AST into the MLIR-style mid-end IR.
//
// Memory model: every user variable (scalar or array) is an addressable
// object - a GlobalVar, a stack object created by memref.alloca, or an array
// parameter (ptr).  Expressions evaluate to scalar SSA temporaries (i32/f32);
// l-values are addressed through `addressOf` (folding constant subscripts
// into byte offsets and emitting multiply-add chains for dynamic ones).
//
// A small scope stack (`env_` + `scopeStack_`) mirrors the block structure of
// the source so shadowing works; `loops_` tracks break/continue targets.
// ---------------------------------------------------------------------------
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../frontend/ASTNode.h"
#include "../../frontend/SemanticAnalysis.h"
#include "../ir/IR.h"

namespace sakura {
namespace ir {

class IRBuilder {
public:
  IRBuilder(CompUnitNode *unit,
            std::unordered_map<std::string, FuncInfo> *funcs,
            const std::string &srcName)
      : unit_(unit), funcs_(funcs) {
    mod_ = std::make_unique<Module>(srcName);
  }

  // Lower the whole unit; returns the IR module.
  std::unique_ptr<Module> run();

private:
  // ---- environment ----
  struct Obj {
    enum class Kind { Global, Slot, ArrayParam };
    Value *addr = nullptr;       // GlobalVar / Alloca / Argument
    Type elem = Type::I32;       // scalar element type
    bool isConst = false;
    bool isArray = false;
    std::vector<int64_t> dims;   // shape (params may carry -1)
  };

  CompUnitNode *unit_;
  std::unordered_map<std::string, FuncInfo> *funcs_;
  std::unique_ptr<Module> mod_;

  Function *curFn_ = nullptr;
  BasicBlock *curBB_ = nullptr;
  Value *retBuf_ = nullptr;    // tensor sret destination of the current function
  bool retIsTensorFn_ = false; // current function returns a tensor
  int tempSeq_ = 0;
  int blockSeq_ = 0;

  std::unordered_map<std::string, Obj *> env_;
  std::vector<std::unique_ptr<Obj>> objArena_;
  struct ScopeFrame {
    std::vector<std::pair<std::string, Obj *>> saved; // null => freshly added
  };
  std::vector<ScopeFrame> scopeStack_;
  struct LoopCtx {
    BasicBlock *cond;   // continue target
    BasicBlock *end;    // break target
  };
  std::vector<LoopCtx> loops_;

  // ---- scopes ----
  void pushScope() { scopeStack_.emplace_back(); }
  void popScope();
  void bind(const std::string &name, Obj *o);
  Obj *lookup(const std::string &name);

  // ---- helpers ----
  Function *findOrDeclareFunc(const std::string &name);   // resolve call target
  Type typeOf(TypeCat t) const {
    switch (t) {
    case TypeCat::FLOAT: return Type::F32;
    case TypeCat::VOID: return Type::Void;
    default: return Type::I32;
    }
  }

  // ---- IR emission primitives ----
  BasicBlock *newBlock();
  Instruction *newInstr(Op op, Type resultTy, int line);
  Value *emitCmp(Cond c, Value *l, Value *r, bool isFloat, int line);
  Value *convert(Value *v, Type dst, int line);   // sitofp / fptosi / id

  ConstantInt *cI(int32_t v) { return mod_->constInt(v); }
  ConstantFloat *cF(float v) { return mod_->constFloat(v); }
  Value *constOf(const ConstVal &cv);

  // ---- module passes ----
  void declareFunctions();
  void emitGlobals();
  void buildFunction(FuncDefNode *fn);
  void finishFunction();

  // ---- statements ----
  void visitStmt(StmtNode *s);
  void visitBlock(BlockStmtNode *b);
  void visitDeclStmt(DeclStmtNode *s);
  void visitIf(IfStmtNode *s);
  void visitWhile(WhileStmtNode *s);
  void visitReturn(ReturnStmtNode *s);
  void visitAssign(AssignStmtNode *s);
  void visitExprStmt(ExprStmtNode *s);

  // variable declarations inside a function
  void emitLocalVar(VarDefNode *def, TypeCat base, bool isConst);
  void emitLocalArrayInit(const std::vector<int64_t> &dims, TypeCat base,
                          InitNode *init, Value *baseAddr);
  void emitArrayZeroFill(Value *baseAddr, int32_t count, Type elem, int line);
  void emitGlobalDef(VarDefNode *def, TypeCat base, bool isConst);

  // ---- expressions ----
  Value *visitExpr(ExprNode *e);       // scalar value (i32 / f32)
  Value *visitLValValue(LValNode *lv); // value of a scalar object
  Value *addressOf(LValNode *lv);      // pointer to object / element / sub-array
  Value *codegenArg(ExprNode *e, const ParamDesc &formal, int line);

  // whole-tensor (TensorType) lowering: tensor values stay first-class at the
  // IR level (tensor.<op> on buffers + shapes), only the back-end expands
  // them into loops.
  void emitTensorExprTo(ExprNode *e, Value *dest);   // write `e` into buffer dest
  void lowerTensorBinary(BinaryExprNode *bn, Value *dest);
  void lowerTensorUnary(UnaryExprNode *un, Value *dest);
  void lowerTensorCall(CallExprNode *cl, Value *dest);
  Value *bufferOfTensor(ExprNode *e);  // address of a whole-tensor value
  Instruction *tensorOp(Op op, Type elem, const std::vector<int64_t> &shape,
                        Value *dest, const std::vector<Value *> &ops, int line);
  void emitTensorCopy(Value *dest, Value *src, Type elem,
                      const std::vector<int64_t> &shape, int line);
  // emit a call (possibly to a tensor-returning function, `sretDest` = hidden
  // result buffer) and return the Call instruction.
  Instruction *lowerCallCodegen(CallExprNode *cl, Value *sretDest);

  // conditions: booleanize, with short-circuit evaluation
  Value *condValue(ExprNode *e);
  Value *evalLogical(BinaryExprNode *e);
  void emitCondBr(ExprNode *cond, BasicBlock *T, BasicBlock *F);

  // ---- IR construction ----
  void emitStore(Value *addr, Value *val, int line);
  Value *emitLoad(Value *addr, Type elem, int line);
  Value *emitAlloca(Type elem, int64_t count, int line);
  Value *emitGep(Value *base, Value *byteOff, int line);
  void emitTerm(Op op, const std::vector<Value *> &ops, int line);
};

} // namespace ir
} // namespace sakura
