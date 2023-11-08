#include <faiss/AutoTune.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
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

int main(int argc, char* argv[]) {
    assert(sizeof(int) == sizeof(float));
    auto start = timeNow(), end = timeNow();
    faiss::IndexIVFFlat* ivf_index;

    std::string use_pre_existing = argv[1];
    size_t M;
    sscanf(argv[2], "%zu", &M);
    size_t nlist;
    sscanf(argv[3], "%zu", &nlist);
    char* training_set_file = argv[4];
    char* base_set_file = argv[5];
    char* index_file = argv[6];
    char* query_file = argv[7];
    char* groundtruth_file = argv[8];
    char* index_params = argv[9];

    if (use_pre_existing == "1") {
        faiss::Index* i = faiss::read_index(index_file);
        ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(i);
        ivf_index->verbose = true;

    } else {
        size_t training_set_dimensions, num_training_vectors;
        start = timeNow();
        float* training_set = fvecs_read(
                training_set_file,
                &training_set_dimensions,
                &num_training_vectors);
        end = timeNow();
        printf("Time taken to read training set: %.4fs\n", elapsed(start, end));

        faiss::IndexHNSWFlat* quantizer =
                new faiss::IndexHNSWFlat(training_set_dimensions, M);
        quantizer->verbose = true;
        ivf_index = new faiss::IndexIVFFlat(
                quantizer, training_set_dimensions, nlist);
        ivf_index->verbose = true;

        // imp
        ivf_index->quantizer_trains_alone = 2;
        ivf_index->own_fields = true;

        start = timeNow();
        ivf_index->train(num_training_vectors, training_set);
        end = timeNow();
        printf("Time taken to train index: %.4fs\n", elapsed(start, end));

        delete[] training_set;

        size_t base_set_dimensions, num_base_vectors;
        float* base_set = fvecs_read(
                base_set_file, &base_set_dimensions, &num_base_vectors);
        end = timeNow();
        assert(training_set_dimensions == base_set_dimensions);
        printf("Time taken to read training set: %.4fs\n", elapsed(start, end));

        start = timeNow();
        ivf_index->add(num_base_vectors, base_set);
        end = timeNow();
        printf("Time taken to add base vectors to the index: %.4fs\n",
               elapsed(start, end));
        delete[] base_set;

        faiss::write_index(ivf_index, index_file);
    }

    size_t query_set_dimensions, num_queries;
    start = timeNow();
    float* query_set =
            fvecs_read(query_file, &query_set_dimensions, &num_queries);
    end = timeNow();
    printf("Time taken to read query set: %.4fs\n", elapsed(start, end));
    size_t num_groundtruth, groundtruth_dimensions;
    start = timeNow();
    int* gt_int = ivecs_read(
            groundtruth_file, &groundtruth_dimensions, &num_groundtruth);
    end = timeNow();
    printf("Time taken to read groundtruth: %.4fs\n", elapsed(start, end));

    faiss::idx_t* groundtruth_set =
            new faiss::idx_t[groundtruth_dimensions * num_groundtruth];
    for (int i = 0; i < groundtruth_dimensions * num_groundtruth; i++) {
        groundtruth_set[i] = gt_int[i];
    }
    delete[] gt_int;

    faiss::idx_t* indices =
            new faiss::idx_t[num_queries * groundtruth_dimensions];
    float* distances = new float[num_queries * groundtruth_dimensions];

    if (std::string(index_params) != "nil") {
        faiss::ParameterSpace().set_index_parameters(ivf_index, index_params);
    }
    start = timeNow();
    ivf_index->search(
            num_queries, query_set, groundtruth_dimensions, distances, indices);
    end = timeNow();
    printf("Time taken to search %d queries: %.4fs\n",
           num_queries,
           elapsed(start, end));

    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (int i = 0; i < num_queries; i++) {
        int gt_nn = groundtruth_set[i * groundtruth_dimensions];
        for (int j = 0; j < groundtruth_dimensions; j++) {
            if (indices[i * groundtruth_dimensions + j] == gt_nn) {
                if (j < 1)
                    n_1++;
                if (j < 10)
                    n_10++;
                if (j < 100)
                    n_100++;
            }
        }
    }
    printf("R@1 = %.4f\n", n_1 / float(num_queries));
    printf("R@10 = %.4f\n", n_10 / float(num_queries));
    printf("R@100 = %.4f\n", n_100 / float(num_queries));
}