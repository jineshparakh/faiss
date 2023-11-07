#include <faiss/AutoTune.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <cassert>
#include <chrono>
#include <iostream>

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed(auto start, auto end) {
    double time_taken =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count();
    time_taken *= 1e-9;
    return time_taken;
}
auto timeNow() {
    return std::chrono::high_resolution_clock::now();
}

int main(){
    size_t training_set_dimensions, num_training_vectors;
    float* training_set = fvecs_read("sift/sift_learn.fvecs", &training_set_dimensions, &num_training_vectors);

    size_t nlist=16;
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(d);
    faiss::IndexIVF* ivf_index = new faiss::IndexIVFFlat(quantizer, training_set_dimensions, nlist);

    ivf_index->train(num_training_vectors, training_set);

    float* centroids = new float[nlist*training_set_dimensions];
    index.quantizer->reconstruct_n(0, nlist, centroids);

    index.search_preassigned()
}