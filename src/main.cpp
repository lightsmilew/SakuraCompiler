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
//   * back-end  : RISCVBackend::run             (pure-cf IR -> .s)
// runMidEndPipeline() (midend/pass/Pipeline.h) owns the PassManager and every
// IR conversion/optimisation pass; the RISCVBackend class (backend/Pipeline.h)
// owns instruction selection, the machine peephole/scheduler, register
// allocation, the post-allocation redundant-move peephole and assembly
// emission.  The driver never assembles passes or calls back-end stages
// itself.  The pipeline hooks print the module at each layer boundary and the
// per-stage pass reports, which back the --dump-affine / --dump-scf /
// --dump-cf / --pass-stats flags (--dump-ir prints all three layer views and
// skips assembly emission).

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "frontend/ASTBuilder.h"
#include "frontend/generate/SysYLexer.h"
#include "frontend/generate/SysYParser.h"

#include "backend/Pipeline.h"
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
  bool dumpFinal = false;    // dump the module right before the back-end
  bool passStats = false;
  sakura::ir::OptLevel optLevel = sakura::ir::OptLevel::O2; // default: on
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--dump-ir") dumpIr = true;
    else if (a == "--dump-affine") dumpAffine = true;
    else if (a == "--dump-scf") dumpScf = true;
    else if (a == "--dump-cf") dumpCf = true;
    else if (a == "--dump-final") dumpFinal = true;
    else if (a == "--pass-stats") passStats = true;
    else if (a == "-O0") optLevel = sakura::ir::OptLevel::O0;
    else if (a == "-O1") optLevel = sakura::ir::OptLevel::O1;
    else if (a == "-O2") optLevel = sakura::ir::OptLevel::O2;
    else if (a == "-o") {
      if (i + 1 < argc) outFile = argv[++i];
    }
    else if (a == "-S") {
      // emit assembly (default behaviour)
    }
    else inputFile = a;
  }
  if (inputFile.empty()) {
    std::cerr << "usage: compiler <input.sy> [-S] [-o out.s] [-O0|-O1|-O2]"
                 " [--dump-ir] [--pass-stats]\n";
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
        },
        optLevel, passStats ? &std::cout : nullptr);

    // ---- back-end: pure-cf IR -> RISC-V assembly -----------------------------
    // The module is guaranteed to contain only cf / func / arith / memref ops
    // (verified inside the mid-end pipeline).  runBackendPipeline runs the
    // whole back-end internally - instruction selection, the block-local
    // peephole + load scheduler (kept off at -O0), register allocation and
    // assembly emission - and returns the final .s text.
    if (dumpFinal)
      std::cout << "// ===== final (pre-ISel) =====\n" << module->toString();
    sakura::backend::BackendOptions bopts;
    bopts.machineOpt = (optLevel != sakura::ir::OptLevel::O0);
    bopts.stats = passStats ? &std::cout : nullptr;
    sakura::backend::RISCVBackend backend;
    std::string asmText = backend.run(module.get(), bopts);

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
