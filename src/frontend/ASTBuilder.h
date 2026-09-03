// ---------------------------------------------------------------------------
// ASTBuilder.h: ANTLR4 visitor that turns the SysY parse tree into the typed
// AST defined in ASTNode.h.
//
// Each visit* override mirrors a grammar rule; expressions are threaded down
// the usual precedence layers and built up again from the leaves.  The result
// is a CompUnitNode with no type information yet - SemanticAnalysis resolves
// types/constants afterwards.
// ---------------------------------------------------------------------------
#pragma once

#include "ASTNode.h"
#include "generate/SysYBaseVisitor.h"

namespace sakura {

// Converts the ANTLR parse tree into our AST.
class ASTBuilder : public SysYBaseVisitor {
public:
  // Entry point. Takes ownership of the returned node.
  std::unique_ptr<CompUnitNode> build(antlr4::tree::ParseTree *tree);

private:
  // helpers
  TypeCat bTypeOf(SysYParser::BTypeContext *b, int line);
  bool bTypeIsTensor(SysYParser::BTypeContext *b);
  ExprNode *visitExpTree(antlr4::tree::ParseTree *t) {
    return std::any_cast<ExprNode *>(visit(t));
  }

  // compilation unit
  std::any visitCompUnit(SysYParser::CompUnitContext *ctx) override;

  // declarations
  std::any visitVariableDeclaration(SysYParser::VariableDeclarationContext *ctx) override;
  std::any visitConstDeclaration(SysYParser::ConstDeclarationContext *ctx) override;
  std::any visitVarDecl(SysYParser::VarDeclContext *ctx) override;
  std::any visitConstDecl(SysYParser::ConstDeclContext *ctx) override;
  std::any visitItemDecl(SysYParser::ItemDeclContext *ctx) override;
  std::any visitItemStmt(SysYParser::ItemStmtContext *ctx) override;

  // functions / params
  std::any visitFuncDef(SysYParser::FuncDefContext *ctx) override;
  std::any visitScalarParam(SysYParser::ScalarParamContext *ctx) override;
  std::any visitArrayParamNoSize(SysYParser::ArrayParamNoSizeContext *ctx) override;
  std::any visitArrayParamWithSize(SysYParser::ArrayParamWithSizeContext *ctx) override;

  // statements
  std::any visitBlock(SysYParser::BlockContext *ctx) override;
  std::any visitAssignStmt(SysYParser::AssignStmtContext *ctx) override;
  std::any visitExprStmt(SysYParser::ExprStmtContext *ctx) override;
  std::any visitBlockStmt(SysYParser::BlockStmtContext *ctx) override;
  std::any visitIfStmt(SysYParser::IfStmtContext *ctx) override;
  std::any visitWhileStmt(SysYParser::WhileStmtContext *ctx) override;
  std::any visitBreakStmt(SysYParser::BreakStmtContext *ctx) override;
  std::any visitContinueStmt(SysYParser::ContinueStmtContext *ctx) override;
  std::any visitReturnStmt(SysYParser::ReturnStmtContext *ctx) override;

  // expressions
  std::any visitExp(SysYParser::ExpContext *ctx) override;
  std::any visitCond(SysYParser::CondContext *ctx) override;
  std::any visitConstExp(SysYParser::ConstExpContext *ctx) override;
  std::any visitLVal(SysYParser::LValContext *ctx) override;

  std::any visitParenExp(SysYParser::ParenExpContext *ctx) override;
  std::any visitLValExp(SysYParser::LValExpContext *ctx) override;
  std::any visitNumberExp(SysYParser::NumberExpContext *ctx) override;
  std::any visitStringLiteralExp(SysYParser::StringLiteralExpContext *ctx) override;
  std::any visitIntNum(SysYParser::IntNumContext *ctx) override;
  std::any visitFloatNum(SysYParser::FloatNumContext *ctx) override;

  std::any visitToPrimaryExp(SysYParser::ToPrimaryExpContext *ctx) override;
  std::any visitCallExp(SysYParser::CallExpContext *ctx) override;
  std::any visitOpUnaryExp(SysYParser::OpUnaryExpContext *ctx) override;
  std::any visitToUnaryExp_mul(SysYParser::ToUnaryExp_mulContext *ctx) override;
  std::any visitMulDivModExp(SysYParser::MulDivModExpContext *ctx) override;
  std::any visitToMulExp_add(SysYParser::ToMulExp_addContext *ctx) override;
  std::any visitAddSubExp(SysYParser::AddSubExpContext *ctx) override;
  std::any visitToAddExp_rel(SysYParser::ToAddExp_relContext *ctx) override;
  std::any visitRelOpExp(SysYParser::RelOpExpContext *ctx) override;
  std::any visitToRelExp_eq(SysYParser::ToRelExp_eqContext *ctx) override;
  std::any visitEqOpExp(SysYParser::EqOpExpContext *ctx) override;
  std::any visitToEqExp_land(SysYParser::ToEqExp_landContext *ctx) override;
  std::any visitLandOpExp(SysYParser::LandOpExpContext *ctx) override;
  std::any visitToLAndExp_lor(SysYParser::ToLAndExp_lorContext *ctx) override;
  std::any visitLorOpExp(SysYParser::LorOpExpContext *ctx) override;

  // initializers
  std::any visitInitExpr(SysYParser::InitExprContext *ctx) override;
  std::any visitInitList(SysYParser::InitListContext *ctx) override;
  std::any visitConstInitExpr(SysYParser::ConstInitExprContext *ctx) override;
  std::any visitConstInitList(SysYParser::ConstInitListContext *ctx) override;

  VarDefNode *buildVarDef(SysYParser::VarDefContext *ctx, bool isConst);
};

} // namespace sakura
