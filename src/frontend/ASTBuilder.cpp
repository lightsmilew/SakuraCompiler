// ASTBuilder implementation: ANTLR4 visitor over the SysY parse tree.
// Grammar tree (BType -> declarations -> statements -> expressions) is
// mirrored by the visit* overrides; numeric literals follow SysY's
// octal/hex/decimal rules and float literals are parsed as single precision.
#include "ASTBuilder.h"

using namespace antlr4;
using namespace sakura;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

TypeCat ASTBuilder::bTypeOf(SysYParser::BTypeContext *b, int line) {
  if (b == nullptr) return TypeCat::VOID;
  // `b` is one of the three labelled alternatives; getText() returns the full
  // phrase ("int", "float", "tensor int", "tensor float").
  std::string t = b->getText();
  if (t.find("float") != std::string::npos) return TypeCat::FLOAT;
  if (t.find("int") != std::string::npos) return TypeCat::INT;
  throw CompileError("invalid base type", line);
}

bool ASTBuilder::bTypeIsTensor(SysYParser::BTypeContext *b) {
  if (b == nullptr) return false;
  // The labelled alternative `tensor (INT|FLOAT)` produces a
  // TypeTensorContext runtime subclass.
  std::string t = b->getText();
  if (t.rfind("tensor", 0) == 0) return true;
  return dynamic_cast<SysYParser::TypeTensorContext *>(b) != nullptr;
}

int64_t parseIntLiteral(const std::string &text) {
  const char *s = text.c_str();
  if (text.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
    return (int64_t)strtoull(s + 2, nullptr, 16);
  if (text.size() > 1 && s[0] == '0')
    return (int64_t)strtoull(s + 1, nullptr, 8);
  return (int64_t)strtoull(s, nullptr, 10);
}

std::unique_ptr<CompUnitNode> ASTBuilder::build(antlr4::tree::ParseTree *tree) {
  std::any r = visit(tree);
  return std::unique_ptr<CompUnitNode>(std::any_cast<CompUnitNode *>(r));
}

std::any ASTBuilder::visitCompUnit(SysYParser::CompUnitContext *ctx) {
  auto unit = new CompUnitNode;
  for (auto d : ctx->decl()) {
    DeclNode *decl = std::any_cast<DeclNode *>(visit(d));
    unit->globals.emplace_back(decl);
  }
  for (auto f : ctx->funcDef()) {
    FuncDefNode *fn = std::any_cast<FuncDefNode *>(visit(f));
    unit->funcs.emplace_back(fn);
  }
  return (std::any)unit;
}

// ---------------------------------------------------------------------------
// declarations
// ---------------------------------------------------------------------------

std::any ASTBuilder::visitVariableDeclaration(SysYParser::VariableDeclarationContext *ctx) {
  return (std::any)std::any_cast<DeclNode *>(visit(ctx->varDecl()));
}

std::any ASTBuilder::visitConstDeclaration(SysYParser::ConstDeclarationContext *ctx) {
  return (std::any)std::any_cast<DeclNode *>(visit(ctx->constDecl()));
}

std::any ASTBuilder::visitVarDecl(SysYParser::VarDeclContext *ctx) {
  auto decl = new DeclNode;
  decl->isConst = false;
  decl->isTensor = bTypeIsTensor(ctx->bType());
  decl->line = ctx->getStart()->getLine();
  decl->base = bTypeOf(ctx->bType(), decl->line);
  for (auto vd : ctx->varDef()) {
    VarDefNode *def = buildVarDef(vd, false);
    decl->defs.emplace_back(def);
  }
  return (std::any)(DeclNode *)decl;
}

std::any ASTBuilder::visitConstDecl(SysYParser::ConstDeclContext *ctx) {
  auto decl = new DeclNode;
  decl->isConst = true;
  decl->isTensor = bTypeIsTensor(ctx->bType());
  decl->line = ctx->getStart()->getLine();
  decl->base = bTypeOf(ctx->bType(), decl->line);
  for (auto cd : ctx->constDef()) {
    auto def = new VarDefNode;
    def->name = cd->Ident()->getText();
    def->line = cd->getStart()->getLine();
    for (auto dim : cd->constExp())
      def->dims.emplace_back(visitExpTree(dim));
    def->init.reset(std::any_cast<InitNode *>(visit(cd->constInitVal())));
    decl->defs.emplace_back(def);
  }
  return (std::any)(DeclNode *)decl;
}

VarDefNode *ASTBuilder::buildVarDef(SysYParser::VarDefContext *ctx, bool isConst) {
  auto def = new VarDefNode;
  def->name = ctx->Ident()->getText();
  def->line = ctx->getStart()->getLine();
  for (auto dim : ctx->constExp())
    def->dims.emplace_back(visitExpTree(dim));
  if (ctx->initVal())
    def->init.reset(std::any_cast<InitNode *>(visit(ctx->initVal())));
  return def;
}

std::any ASTBuilder::visitItemDecl(SysYParser::ItemDeclContext *ctx) {
  auto stmt = new DeclStmtNode;
  stmt->line = ctx->getStart()->getLine();
  DeclNode *decl = std::any_cast<DeclNode *>(visit(ctx->decl()));
  stmt->decl.reset(decl);
  return (std::any)(StmtNode *)stmt;
}

std::any ASTBuilder::visitItemStmt(SysYParser::ItemStmtContext *ctx) {
  return (std::any)std::any_cast<StmtNode *>(visit(ctx->stmt()));
}

// ---------------------------------------------------------------------------
// initializers
// ---------------------------------------------------------------------------

std::any ASTBuilder::visitInitExpr(SysYParser::InitExprContext *ctx) {
  auto init = new InitNode;
  init->kind = InitNode::EXPR;
  init->expr.reset(visitExpTree(ctx->exp()));
  return (std::any)(InitNode *)init;
}

std::any ASTBuilder::visitInitList(SysYParser::InitListContext *ctx) {
  auto init = new InitNode;
  init->kind = InitNode::LIST;
  for (auto child : ctx->initVal())
    init->list.emplace_back(std::any_cast<InitNode *>(visit(child)));
  return (std::any)(InitNode *)init;
}

std::any ASTBuilder::visitConstInitExpr(SysYParser::ConstInitExprContext *ctx) {
  auto init = new InitNode;
  init->kind = InitNode::EXPR;
  init->expr.reset(visitExpTree(ctx->constExp()));
  return (std::any)(InitNode *)init;
}

std::any ASTBuilder::visitConstInitList(SysYParser::ConstInitListContext *ctx) {
  auto init = new InitNode;
  init->kind = InitNode::LIST;
  for (auto child : ctx->constInitVal())
    init->list.emplace_back(std::any_cast<InitNode *>(visit(child)));
  return (std::any)(InitNode *)init;
}

// ---------------------------------------------------------------------------
// functions
// ---------------------------------------------------------------------------

std::any ASTBuilder::visitFuncDef(SysYParser::FuncDefContext *ctx) {
  auto fn = new FuncDefNode;
  fn->line = ctx->getStart()->getLine();
  fn->name = ctx->Ident()->getText();

  SysYParser::FuncTypeContext *ft = ctx->funcType();
  if (dynamic_cast<SysYParser::TypeVoidContext *>(ft)) {
    fn->retType = TypeCat::VOID;
  } else {
    SysYParser::TypeBTypeContext *btc =
        dynamic_cast<SysYParser::TypeBTypeContext *>(ft);
    if (!btc) throw CompileError("internal: malformed function type", fn->line);
    SysYParser::BTypeContext *bt = btc->bType();
    fn->retType = bTypeOf(bt, fn->line);
    fn->retIsTensor = bTypeIsTensor(bt);
  }
  if (ctx->funcFParams()) {
    for (auto fp : ctx->funcFParams()->funcFParam())
      fn->params.emplace_back(std::any_cast<ParamNode *>(visit(fp)));
  }
  fn->body.reset(std::any_cast<BlockStmtNode *>(visit(ctx->block())));
  return (std::any)(FuncDefNode *)fn;
}

std::any ASTBuilder::visitScalarParam(SysYParser::ScalarParamContext *ctx) {
  auto p = new ParamNode;
  p->line = ctx->getStart()->getLine();
  p->name = ctx->Ident()->getText();
  p->isArray = false;
  p->type.base = bTypeOf(ctx->bType(), p->line);
  p->type.isTensor = bTypeIsTensor(ctx->bType());
  return (std::any)(ParamNode *)p;
}

std::any ASTBuilder::visitArrayParamNoSize(SysYParser::ArrayParamNoSizeContext *ctx) {
  auto p = new ParamNode;
  p->line = ctx->getStart()->getLine();
  p->name = ctx->Ident()->getText();
  p->isArray = true;
  p->firstDimOmitted = true;
  p->type.base = bTypeOf(ctx->bType(), p->line);
  p->type.isTensor = bTypeIsTensor(ctx->bType());
  for (auto dim : ctx->constExp())
    p->dimExprs.emplace_back(visitExpTree(dim));
  return (std::any)(ParamNode *)p;
}

std::any ASTBuilder::visitArrayParamWithSize(SysYParser::ArrayParamWithSizeContext *ctx) {
  auto p = new ParamNode;
  p->line = ctx->getStart()->getLine();
  p->name = ctx->Ident()->getText();
  p->isArray = true;
  p->firstDimOmitted = false;
  p->type.base = bTypeOf(ctx->bType(), p->line);
  p->type.isTensor = bTypeIsTensor(ctx->bType());
  for (auto dim : ctx->constExp())
    p->dimExprs.emplace_back(visitExpTree(dim));
  return (std::any)(ParamNode *)p;
}

// ---------------------------------------------------------------------------
// statements
// ---------------------------------------------------------------------------

std::any ASTBuilder::visitBlock(SysYParser::BlockContext *ctx) {
  auto blk = new BlockStmtNode;
  blk->line = ctx->getStart()->getLine();
  for (auto item : ctx->blockItem())
    blk->body.emplace_back(std::any_cast<StmtNode *>(visit(item)));
  return (std::any)(BlockStmtNode *)blk;
}

std::any ASTBuilder::visitAssignStmt(SysYParser::AssignStmtContext *ctx) {
  auto s = new AssignStmtNode;
  s->line = ctx->getStart()->getLine();
  s->lhs.reset(visitExpTree(ctx->lVal()));
  s->rhs.reset(visitExpTree(ctx->exp()));
  return (std::any)(StmtNode *)s;
}

std::any ASTBuilder::visitExprStmt(SysYParser::ExprStmtContext *ctx) {
  auto s = new ExprStmtNode;
  s->line = ctx->getStart()->getLine();
  if (ctx->exp()) s->expr.reset(visitExpTree(ctx->exp()));
  return (std::any)(StmtNode *)s;
}

std::any ASTBuilder::visitBlockStmt(SysYParser::BlockStmtContext *ctx) {
  return (std::any)(StmtNode *)std::any_cast<BlockStmtNode *>(visit(ctx->block()));
}

std::any ASTBuilder::visitIfStmt(SysYParser::IfStmtContext *ctx) {
  auto s = new IfStmtNode;
  s->line = ctx->getStart()->getLine();
  s->cond.reset(visitExpTree(ctx->cond()));
  auto stmts = ctx->stmt();
  s->thenStmt.reset(std::any_cast<StmtNode *>(visit(stmts[0])));
  if (stmts.size() > 1)
    s->elseStmt.reset(std::any_cast<StmtNode *>(visit(stmts[1])));
  return (std::any)(StmtNode *)s;
}

std::any ASTBuilder::visitWhileStmt(SysYParser::WhileStmtContext *ctx) {
  auto s = new WhileStmtNode;
  s->line = ctx->getStart()->getLine();
  s->cond.reset(visitExpTree(ctx->cond()));
  s->body.reset(std::any_cast<StmtNode *>(visit(ctx->stmt())));
  return (std::any)(StmtNode *)s;
}

std::any ASTBuilder::visitBreakStmt(SysYParser::BreakStmtContext *ctx) {
  auto s = new BreakStmtNode;
  s->line = ctx->getStart()->getLine();
  return (std::any)(StmtNode *)s;
}

std::any ASTBuilder::visitContinueStmt(SysYParser::ContinueStmtContext *ctx) {
  auto s = new ContinueStmtNode;
  s->line = ctx->getStart()->getLine();
  return (std::any)(StmtNode *)s;
}

std::any ASTBuilder::visitReturnStmt(SysYParser::ReturnStmtContext *ctx) {
  auto s = new ReturnStmtNode;
  s->line = ctx->getStart()->getLine();
  if (ctx->exp()) s->value.reset(visitExpTree(ctx->exp()));
  return (std::any)(StmtNode *)s;
}

// ---------------------------------------------------------------------------
// expressions
// ---------------------------------------------------------------------------

std::any ASTBuilder::visitExp(SysYParser::ExpContext *ctx) {
  return (std::any)visitExpTree(ctx->lOrExp());
}
std::any ASTBuilder::visitCond(SysYParser::CondContext *ctx) {
  return (std::any)visitExpTree(ctx->lOrExp());
}
std::any ASTBuilder::visitConstExp(SysYParser::ConstExpContext *ctx) {
  return (std::any)visitExpTree(ctx->addExp());
}

std::any ASTBuilder::visitLVal(SysYParser::LValContext *ctx) {
  auto e = new LValNode;
  e->line = ctx->getStart()->getLine();
  e->name = ctx->Ident()->getText();
  for (auto idx : ctx->exp())
    e->indices.emplace_back(visitExpTree(idx));
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitParenExp(SysYParser::ParenExpContext *ctx) {
  return (std::any)visitExpTree(ctx->exp());
}
std::any ASTBuilder::visitLValExp(SysYParser::LValExpContext *ctx) {
  return (std::any)visitExpTree(ctx->lVal());
}
std::any ASTBuilder::visitNumberExp(SysYParser::NumberExpContext *ctx) {
  return (std::any)visitExpTree(ctx->number());
}
std::any ASTBuilder::visitStringLiteralExp(SysYParser::StringLiteralExpContext *ctx) {
  auto e = new StringLitNode;
  e->line = ctx->getStart()->getLine();
  std::string t = ctx->STRING_LITERAL()->getText();
  if (t.size() >= 2) t = t.substr(1, t.size() - 2);
  e->content = t;
  return (std::any)(ExprNode *)e;
}
std::any ASTBuilder::visitIntNum(SysYParser::IntNumContext *ctx) {
  auto e = new IntLitNode;
  e->line = ctx->getStart()->getLine();
  e->value = parseIntLiteral(ctx->IntConst()->getText());
  return (std::any)(ExprNode *)e;
}
std::any ASTBuilder::visitFloatNum(SysYParser::FloatNumContext *ctx) {
  auto e = new FloatLitNode;
  e->line = ctx->getStart()->getLine();
  e->text = ctx->FloatConst()->getText();
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitToPrimaryExp(SysYParser::ToPrimaryExpContext *ctx) {
  return (std::any)visitExpTree(ctx->primaryExp());
}

std::any ASTBuilder::visitCallExp(SysYParser::CallExpContext *ctx) {
  auto e = new CallExprNode;
  e->line = ctx->getStart()->getLine();
  e->name = ctx->Ident()->getText();
  if (ctx->funcRParams()) {
    for (auto a : ctx->funcRParams()->exp())
      e->args.emplace_back(visitExpTree(a));
  }
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitOpUnaryExp(SysYParser::OpUnaryExpContext *ctx) {
  auto e = new UnaryExprNode;
  e->line = ctx->getStart()->getLine();
  std::string uop = ctx->unaryOp()->getText();
  if (uop == "+") e->op = UnaryOp::PLUS;
  else if (uop == "-") e->op = UnaryOp::MINUS;
  else e->op = UnaryOp::NOT;
  e->operand.reset(visitExpTree(ctx->unaryExp()));
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitToUnaryExp_mul(SysYParser::ToUnaryExp_mulContext *ctx) {
  return (std::any)visitExpTree(ctx->unaryExp());
}

std::any ASTBuilder::visitMulDivModExp(SysYParser::MulDivModExpContext *ctx) {
  auto e = new BinaryExprNode;
  e->line = ctx->getStart()->getLine();
  e->lhs.reset(visitExpTree(ctx->mulExp()));
  e->rhs.reset(visitExpTree(ctx->unaryExp()));
  if (ctx->MUL()) e->op = BinaryOp::MUL;
  else if (ctx->DIV()) e->op = BinaryOp::DIV;
  else if (ctx->MOD()) e->op = BinaryOp::MOD;
  else e->op = BinaryOp::MATMUL; // '@'
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitToMulExp_add(SysYParser::ToMulExp_addContext *ctx) {
  return (std::any)visitExpTree(ctx->mulExp());
}

std::any ASTBuilder::visitAddSubExp(SysYParser::AddSubExpContext *ctx) {
  auto e = new BinaryExprNode;
  e->line = ctx->getStart()->getLine();
  e->lhs.reset(visitExpTree(ctx->addExp()));
  e->rhs.reset(visitExpTree(ctx->mulExp()));
  e->op = ctx->PLUS() ? BinaryOp::ADD : BinaryOp::SUB;
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitToAddExp_rel(SysYParser::ToAddExp_relContext *ctx) {
  return (std::any)visitExpTree(ctx->addExp());
}

std::any ASTBuilder::visitRelOpExp(SysYParser::RelOpExpContext *ctx) {
  auto e = new BinaryExprNode;
  e->line = ctx->getStart()->getLine();
  e->lhs.reset(visitExpTree(ctx->relExp()));
  e->rhs.reset(visitExpTree(ctx->addExp()));
  if (ctx->LT()) e->op = BinaryOp::LT;
  else if (ctx->GT()) e->op = BinaryOp::GT;
  else if (ctx->LE()) e->op = BinaryOp::LE;
  else e->op = BinaryOp::GE;
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitToRelExp_eq(SysYParser::ToRelExp_eqContext *ctx) {
  return (std::any)visitExpTree(ctx->relExp());
}

std::any ASTBuilder::visitEqOpExp(SysYParser::EqOpExpContext *ctx) {
  auto e = new BinaryExprNode;
  e->line = ctx->getStart()->getLine();
  e->lhs.reset(visitExpTree(ctx->eqExp()));
  e->rhs.reset(visitExpTree(ctx->relExp()));
  e->op = ctx->EQ() ? BinaryOp::EQ : BinaryOp::NE;
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitToEqExp_land(SysYParser::ToEqExp_landContext *ctx) {
  return (std::any)visitExpTree(ctx->eqExp());
}

std::any ASTBuilder::visitLandOpExp(SysYParser::LandOpExpContext *ctx) {
  auto e = new BinaryExprNode;
  e->line = ctx->getStart()->getLine();
  e->lhs.reset(visitExpTree(ctx->lAndExp()));
  e->rhs.reset(visitExpTree(ctx->eqExp()));
  e->op = BinaryOp::AND;
  return (std::any)(ExprNode *)e;
}

std::any ASTBuilder::visitToLAndExp_lor(SysYParser::ToLAndExp_lorContext *ctx) {
  return (std::any)visitExpTree(ctx->lAndExp());
}

std::any ASTBuilder::visitLorOpExp(SysYParser::LorOpExpContext *ctx) {
  auto e = new BinaryExprNode;
  e->line = ctx->getStart()->getLine();
  e->lhs.reset(visitExpTree(ctx->lOrExp()));
  e->rhs.reset(visitExpTree(ctx->lAndExp()));
  e->op = BinaryOp::OR;
  return (std::any)(ExprNode *)e;
}
