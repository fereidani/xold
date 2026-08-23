/* A tiny object with global, static, and external symbols plus relocations. */
int global_counter = 0;
static int hidden = 42;

extern int external_func(void);

int local_add(int a, int b) {
    return a + b + hidden;
}

int entry(void) {
    global_counter++;
    return local_add(1, 2) + external_func();
}
