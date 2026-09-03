// ---------------------------------------------------------------------------
// ASTNode.h: the typed abstract-syntax tree produced by ASTBuilder.
//
// Grammar coverage (SysY):
//   * expressions : literals, l-values with arbitrary subscripts, unary
//     (+/-/!), binary arithmetic/comparison/logical ops, calls;
//   * statements  : blocks, declarations, assignment, expression stmts,
//     if/else, while, break/continue, return;
//   * declarations: const / non-const variables and arrays (with initialiser
//     brace lists), function parameters and definitions.
//
// Several nodes carry *analysis results* that SemanticAnalysis fills in and
// IRBuilder later consumes (ExprNode::evalType / cval / isArrayExpr and the
// LValNode resolution fields, VarDefNode::type, ParamNode::type ...).
// ---------------------------------------------------------------------------
#pragma once

#include "../common/Common.h"

namespace sakura {

// ---- operators (source spelling via unaryOpName / binaryOpName) -----------
enum class UnaryOp { PLUS, MINUS, NOT };
enum class BinaryOp {
  ADD, SUB, MUL, DIV, MOD, MATMUL,
  LT, GT, LE, GE, EQ, NE, AND, OR
};

inline const char *unaryOpName(UnaryOp op) {
  switch (op) {
  case UnaryOp::PLUS: return "+";
  case UnaryOp::MINUS: return "-";
  case UnaryOp::NOT: return "!";
  }
  return "?";
}
inline const char *binaryOpName(BinaryOp op) {
  switch (op) {
  case BinaryOp::ADD: return "+";
  case BinaryOp::SUB: return "-";
  case BinaryOp::MUL: return "*";
  case BinaryOp::DIV: return "/";
  case BinaryOp::MOD: return "%";
  case BinaryOp::MATMUL: return "@";
  case BinaryOp::LT: return "<";
  case BinaryOp::GT: return ">";
  case BinaryOp::LE: return "<=";
  case BinaryOp::GE: return ">=";
  case BinaryOp::EQ: return "==";
  case BinaryOp::NE: return "!=";
  case BinaryOp::AND: return "&&";
  case BinaryOp::OR: return "||";
  }
  return "?";
}

// ---- The variable "shape" type used by the front-end (arrays / tensors). --
struct VarType {
  TypeCat base = TypeCat::VOID;
  bool isConst = false;
  bool isTensor = false;          // `tensor int a[N]` vs a plain SysY array
  std::vector<int64_t> dims;  // empty => scalar. For array parameters the
                              // first dimension may be -1 ("unknown").
  bool isArray() const { return !dims.empty(); }
  int64_t numElements() const {
    int64_t n = 1;
    for (int64_t d : dims) {
      if (d < 0) continue;
      n *= d;
    }
    return n;
  }
};

// ---- Expression / lvalue info filled in by the semantic analyser ----
enum class LValKind : uint8_t { NONE, GLOBAL_VAR, LOCAL_VAR, PARAM };

struct ExprNode;
struct StmtNode;
struct VarDefNode;

// ---- Forward declarations ----
struct FuncDefNode;

// ---- Abstract expression ----
struct ExprNode {
  virtual ~ExprNode() = default;
  // Filled by SemanticAnalysis:
  TypeCat evalType = TypeCat::INT; // result scalar type (INT/FLOAT); VOID used for
                                   // calls to void functions / strings.
  bool isLValue = false;           // addressable scalar / sub-array object
  ConstVal cval;                   // valid => compile-time constant
  bool isArrayExpr = false;        // this expression denotes an array object
  bool isTensorVal = false;        // expression value is a *whole tensor*
  std::vector<int64_t> tShape;     // static shape when isTensorVal
  int line = 0;
};

// lVal: Ident ( '[' exp ']' )*
struct LValNode : ExprNode {
  std::string name;
  std::vector<std::unique_ptr<ExprNode>> indices;
  // Semantic results:
  std::vector<int64_t> varDims;   // shape of the resolved variable/parameter
  LValKind varKind = LValKind::NONE;
  bool varIsConst = false;
  bool varIsTensor = false;       // resolves to a `tensor` object / param
};

struct IntLitNode : ExprNode {
  int64_t value = 0; // parsed integer (kept 64-bit; semantic narrows when needed)
};

struct FloatLitNode : ExprNode {
  std::string text; // literal source text, parsed by semantic
};

struct StringLitNode : ExprNode {
  std::string content; // raw contents between quotes (uninterpreted)
};

struct UnaryExprNode : ExprNode {
  UnaryOp op = UnaryOp::PLUS;
  std::unique_ptr<ExprNode> operand;
};

struct BinaryExprNode : ExprNode {
  BinaryOp op = BinaryOp::ADD;
  std::unique_ptr<ExprNode> lhs;
  std::unique_ptr<ExprNode> rhs;
};

struct CallExprNode : ExprNode {
  std::string name;
  std::vector<std::unique_ptr<ExprNode>> args;
  // Semantic: if this is a user function, retType/evalType describe it.
};

// ---- Statements ----
struct StmtNode {
  virtual ~StmtNode() = default;
  int line = 0;
};

// Initializer: either an expression or a brace-enclosed list.
struct InitNode {
  enum Kind { EXPR, LIST };
  Kind kind = EXPR;
  std::unique_ptr<ExprNode> expr;         // when EXPR
  std::vector<std::unique_ptr<InitNode>> list; // when LIST
};

struct VarDefNode {
  std::string name;
  std::vector<std::unique_ptr<ExprNode>> dims; // const expressions
  std::unique_ptr<InitNode> init;              // may be null
  VarType type; // filled by semantic (dims resolved)
  int line = 0;
};

// A declaration statement (appears at global scope or inside a block).
struct DeclNode {
  bool isConst = false;
  bool isTensor = false;        // `tensor int` / `tensor float`
  TypeCat base = TypeCat::INT; // INT / FLOAT (element type)
  std::vector<std::unique_ptr<VarDefNode>> defs;
  int line = 0;
};

struct BlockStmtNode : StmtNode {
  std::vector<std::unique_ptr<StmtNode>> body; // DeclStmtNode or other stmts
};

struct DeclStmtNode : StmtNode {
  std::unique_ptr<DeclNode> decl;
};

struct AssignStmtNode : StmtNode {
  std::unique_ptr<ExprNode> lhs;
  std::unique_ptr<ExprNode> rhs;
};

struct ExprStmtNode : StmtNode {
  std::unique_ptr<ExprNode> expr; // may be null for empty ';'
};

struct IfStmtNode : StmtNode {
  std::unique_ptr<ExprNode> cond;
  std::unique_ptr<StmtNode> thenStmt;
  std::unique_ptr<StmtNode> elseStmt;
};

struct WhileStmtNode : StmtNode {
  std::unique_ptr<ExprNode> cond;
  std::unique_ptr<StmtNode> body;
};

struct BreakStmtNode : StmtNode {};
struct ContinueStmtNode : StmtNode {};

struct ReturnStmtNode : StmtNode {
  std::unique_ptr<ExprNode> value; // null => bare return;
};

// ---- Functions & compilation unit ----
struct ParamNode {
  std::string name;
  bool isArray = false;
  bool firstDimOmitted = false; // `int a[]` style
  std::vector<std::unique_ptr<ExprNode>> dimExprs; // explicit dimension sizes
  VarType type; // resolved by semantic analysis
  int line = 0;
};

struct FuncDefNode {
  std::string name;
  TypeCat retType = TypeCat::VOID;
  bool retIsTensor = false;             // function returns a tensor
  std::vector<int64_t> retDims;         // inferred static return shape
  std::vector<std::unique_ptr<ParamNode>> params;
  std::unique_ptr<BlockStmtNode> body;
  int line = 0;
  bool isBuiltin = false; // library function (no body)
};

struct CompUnitNode {
  std::vector<std::unique_ptr<DeclNode>> globals;
  std::vector<std::unique_ptr<FuncDefNode>> funcs;
};

} // namespace sakura
