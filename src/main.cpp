// SakuraCompiler driver.
// Usage: compiler <input.sy> [-S] [-o out.s] [--dump-ir] [--dump-affine]
//        [--dump-scf] [--dump-cf]
// Runs the full pipeline: lex/parse (ANTLR4) -> AST + semantic analysis ->
// MLIR-style mid-end IR -> RISC-V instruction selection -> graph-colouring
// register allocation -> assembly emission.
//
// main only wires the three high-level stages together:
//   * front-end : ASTBuilder + SemanticAnalysis  (input.sy -> typed AST)
//   * mid-end   : IRBuilder -> runMidEndPipeline (AST -> pure-cf IR)
//   * back-end  : ISel -> RA -> Asm              (pure-cf IR -> .s)
// runMidEndPipeline() (midend/pass/Pipeline.h) owns the PassManager and every
// IR conversion/optimisation pass; the driver never assembles passes itself.
// Its LayerView hook prints the module at each layer boundary, which backs
// the --dump-affine / --dump-scf / --dump-cf flags (--dump-ir prints all
// three and skips assembly emission).

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "frontend/ASTBuilder.h"
#include "frontend/generate/SysYLexer.h"
#include "frontend/generate/SysYParser.h"

#include "backend/Asm.h"
#include "backend/ISel.h"
#include "backend/RA.h"
#include "frontend/SemanticAnalysis.h"
#include "midend/irbuild/IRBuilder.h"
#include "midend/pass/Pipeline.h"

using namespace antlr4;
using namespace sakura;

namespace {

// Error listener that throws on the first syntax error so that we can report a
// clean single error and exit.
struct Thrower : public BaseErrorListener {
  void syntaxError(Recognizer *, Token *, size_t line, size_t charPositionInLine,
                   const std::string &msg, std::exception_ptr) override {
    throw CompileError("syntax error at line " + std::to_string(line) + ", column " +
                           std::to_string(charPositionInLine + 1) + ": " + msg,
                       (int)line);
  }
};

std::string readAll(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw CompileError("cannot open input file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

} // namespace

int main(int argc, char **argv) {
  std::string inputFile;
  std::string outFile;
  bool dumpIr = false;       // dump all three layer views, suppress assembly
  bool dumpAffine = false, dumpScf = false, dumpCf = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--dump-ir") dumpIr = true;
    else if (a == "--dump-affine") dumpAffine = true;
    else if (a == "--dump-scf") dumpScf = true;
    else if (a == "--dump-cf") dumpCf = true;
    else if (a == "-o") {
      if (i + 1 < argc) outFile = argv[++i];
    }
    else if (a == "-S") {
      // emit assembly (default behaviour)
    }
    else inputFile = a;
  }
  if (inputFile.empty()) {
    std::cerr << "usage: compiler <input.sy> [-S] [-o out.s] [--dump-ir]\n";
    return 1;
  }

  try {
    std::string text = readAll(inputFile);

    ANTLRInputStream stream(text);
    SysYLexer lexer(&stream);
    lexer.removeErrorListeners();
    lexer.addErrorListener(new Thrower());
    CommonTokenStream tokens(&lexer);
    SysYParser parser(&tokens);
    parser.removeErrorListeners();
    parser.addErrorListener(new Thrower());
    tree::ParseTree *tree = parser.compUnit();

    ASTBuilder builder;
    auto unit = builder.build(tree);

    // Semantic analysis: resolve scopes / types / constant values and report
    // semantic errors.
    SemanticAnalyzer analyzer(unit.get());
    analyzer.analyze();

    // ---- mid-end: AST -> pure-cf IR ------------------------------------------
    // Lower the checked AST into the MLIR-style IR module and run the whole
    // mid-end pipeline in one call: IRBuilder produces the affine layer, the
    // pipeline (internally a PassManager) lowers it through the scf layer to
    // flat cf and verifies the result.  `view` prints the module whenever the
    // pipeline enters a layer, giving --dump-affine / --dump-scf / --dump-cf.
    auto base = inputFile.substr(inputFile.find_last_of("/\\") + 1);
    sakura::ir::IRBuilder irb(unit.get(), &analyzer.functions(), base);
    auto module = irb.run();
    sakura::ir::runMidEndPipeline(
        *module, [&](const char *layer, sakura::ir::Module &m) {
          std::string L = layer;
          bool show = dumpIr;
          show = show || (dumpAffine && L == "affine") ||
                 (dumpScf && L == "scf") || (dumpCf && L == "cf");
          if (show)
            std::cout << "// ===== " << layer << " layer =====\n"
                      << m.toString();
        });

    // ---- back-end: pure-cf IR -> RISC-V assembly -----------------------------
    // The module is guaranteed to contain only cf / func / arith / memref ops
    // (verified inside the mid-end pipeline).  Instruction selection + graph-
    // colouring register allocation + assembly emission.
    auto fns = sakura::backend::runInstructionSelection(module.get());
    sakura::backend::runRegisterAlloc(fns);
    std::string asmText = sakura::backend::emitAssembly(module.get(), fns);

    std::ostream *os = &std::cout;
    std::ofstream ofs;
    if (!outFile.empty()) {
      ofs.open(outFile, std::ios::binary);
      if (!ofs) throw CompileError("cannot open output file: " + outFile);
      os = &ofs;
    }
    if (!dumpIr) *os << asmText;
    return 0;
  } catch (CompileError &e) {
    std::cerr << "Error" << (e.line > 0 ? " at line " + std::to_string(e.line) : "")
              << ": " << e.what() << "\n";
    return 1;
  } catch (std::exception &e) {
    std::cerr << "internal error: " << e.what() << "\n";
    return 1;
  }
}
