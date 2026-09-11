// Minimal SysY runtime declarations, so a .sy translation unit can be fed to
// a stock C/C++ compiler (clang -x c++) for reference-codegen comparisons.
// The SysY library exports the timers under their `_sysy_` names; the language
// spelling is the unprefixed one, so bind them with macros.
extern "C" {
int getint();
float getfloat();
int getch();
void putint(int);
void putfloat(float);
void putch(int);
// `void*` so the multidimensional array argument (`getarray(A[i])` with
// `int A[1400][1400]`) is accepted without an implicit-conversion error.
int getarray(void *);
void putarray(int, void *);
void _sysy_starttime();
void _sysy_stoptime();
}

#define starttime() _sysy_starttime()
#define stoptime() _sysy_stoptime()
