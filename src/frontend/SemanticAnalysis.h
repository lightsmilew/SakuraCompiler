// ---------------------------------------------------------------------------
// SemanticAnalysis.h: name resolution, type checking and constant folding.
//
// Walks the AST produced by ASTBuilder and annotates it in place:
//   * scopes are pushed/popped for blocks and functions;
//   * every l-value resolves to a VarSymbol whose dims / const-ness are
//     recorded back onto the LValNode for IRBuilder;
//   * expression result types (evalType), lvalue-ness and compile-time
//     constant values (cval) are computed, including full const-initialiser
//     folding through flattenConstInit() / convertConst();
//   * function signatures (user + SysY library) are collected for IRBuilder.
// Any violation raises a CompileError carrying the offending source line.
// ---------------------------------------------------------------------------
#pragma once

#include "ASTNode.h"
#include <unordered_map>

namespace sakura {

// A variable/parameter symbol resolved by the semantic analyser.
struct VarSymbol {
  std::string name;
  VarType type;                 // resolved dims (array params may hold -1)
  bool isConst = false;
  bool isGlobal = false;
  bool isParam = false;
  bool hasInit = false;
  ConstVal value;               // value of a const scalar (INT/FLOAT base)
  bool hasFlat = false;         // const array => flat[0..numElem)
  std::vector<ConstVal> flat;
};

// Signature of a user or library function.
struct FuncInfo {
  TypeCat ret = TypeCat::VOID;
  bool retIsTensor = false;         // function returns a tensor (via sret)
  std::vector<int64_t> retDims;     // inferred static tensor return shape
  std::vector<VarType> params;  // array params keep their dims
  FuncDefNode *def = nullptr;   // null => library function
  bool builtin = false;
};

// Flatten a const initializer tree into one value per element of an object
// with shape `dims` and element scalar type `base`.  All leaf expressions are
// required to already carry a valid compile-time constant.  Missing elements
// are zero filled; too many elements raise a CompileError at `line`.
void flattenConstInit(const std::vector<int64_t> &dims, TypeCat base,
                      InitNode *init, std::vector<ConstVal> &out, int line);

// Convert a compile-time constant to the destination scalar type using the
// same truncation rules as the runtime instructions (fcvt.w.s etc.).
ConstVal convertConst(TypeCat to, const ConstVal &v);

class SemanticAnalyzer {
public:
  SemanticAnalyzer(CompUnitNode *unit) : unit_(unit) {}

  // Runs the whole analysis; throws CompileError on the first semantic error.
  void analyze();

  // Result tables (valid after analyze()).
  std::unordered_map<std::string, FuncInfo> &functions() { return funcs_; }

private:
  CompUnitNode *unit_;
  std::unordered_map<std::string, FuncInfo> funcs_;

  // ---- scopes ----
  std::vector<std::unordered_map<std::string, VarSymbol *>> scopes_;
  std::vector<std::unique_ptr<VarSymbol>> arena_;
  VarSymbol *resolve(const std::string &name);
  VarSymbol *declare(const std::string &name, VarType type, bool isConst,
                     bool isGlobal, bool isParam, int line);
  void pushScope() { scopes_.emplace_back(); }
  void popScope() { scopes_.pop_back(); }

  // ---- current function context ----
  FuncInfo *curFunc_ = nullptr;
  FuncDefNode *curFnNode_ = nullptr;
  int loopDepth_ = 0;

  void error(const std::string &msg, int line);

  // per-node annotation helpers
  void annotateScalar(ExprNode *e, TypeCat t, const ConstVal &v = ConstVal());
  void requireScalar(ExprNode *e);
  void requireConst(ExprNode *e, const std::string &what);

  // main passes
  void analyzeGlobal(DeclNode *decl);
  void checkGlobalDef(VarDefNode *def, bool isConst, TypeCat base,
                      bool isTensor);
  void checkConstInitTree(TypeCat base, InitNode *init, int line);
  void checkLocalDef(VarDefNode *def, bool isConst, TypeCat base,
                     bool isTensor);
  void checkLocalInitTree(TypeCat base, InitNode *init, int line);
  void resolveParamType(ParamNode *p);
  void registerFuncSignatures();
  void analyzeFuncBody(FuncDefNode *fn);

  // statements & expressions
  void checkStmt(StmtNode *s);
  void checkBlock(BlockStmtNode *b);
  void checkDeclStmt(DeclStmtNode *s);
  void checkCond(ExprNode *e);
  void checkExpr(ExprNode *e, bool asRvalue, bool allowArrayResult);
  void checkCall(CallExprNode *e);
  void checkBinary(BinaryExprNode *e);
  void checkUnary(UnaryExprNode *e);
  void checkLValExpr(LValNode *e, bool asRvalue);
  void checkReturn(ReturnStmtNode *s);
  void checkAssign(AssignStmtNode *s);

  // tensor (TensorType) semantics
  void checkTensorUnary(UnaryExprNode *e);        // +x / -x on tensors
  void checkTensorBinary(BinaryExprNode *e);      // tensor/scalar promotion rules
  void checkTensorAssign(AssignStmtNode *s);      // whole-tensor l-value stores
  bool tensorShapeMatches(const VarType &param, const ExprNode *arg) const;
  void setTensorResult(ExprNode *e, TypeCat base,
                       const std::vector<int64_t> &shape);

  // constant folding of an already checked expression (conservative)
  void foldExpr(ExprNode *e);
  void foldBinary(BinaryExprNode *e);
  void foldUnary(UnaryExprNode *e);
};

} // namespace sakura
