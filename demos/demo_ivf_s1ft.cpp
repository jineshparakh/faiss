// #include<faiss/IndexIVFFlat.h>
// #include<faiss/IndexFlat.h>
// #include<faiss/MetricType.h>
// #include<faiss/Clustering.h>
// #include <faiss/index_factory.h>
// #include<faiss/IndexHNSW.h>
// #include <faiss/index_io.h>
// #include<faiss/index_factory.h>

// #include<vector>
// #include<sys/stat.h>
// #include<iostream>
// #include<iomanip>

// #include <sys/time.h>


// float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
//     FILE* f = fopen(fname, "r");
//     if (!f) {
//         fprintf(stderr, "could not open %s\n", fname);
//         perror("");
//         abort();
//     }
//     int d;
//     fread(&d, 1, sizeof(int), f);
//     assert((d > 0 && d < 1000000) || !"unreasonable dimension");
//     fseek(f, 0, SEEK_SET);
//     struct stat st;
//     fstat(fileno(f), &st);
//     size_t sz = st.st_size;
//     assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
//     size_t n = sz / ((d + 1) * 4);

//     *d_out = d;
//     *n_out = n;
//     float* x = new float[n * (d + 1)];
//     size_t nr = fread(x, sizeof(float), n * (d + 1), f);
//     assert(nr == n * (d + 1) || !"could not read whole file");

//     // shift array to remove row headers
//     for (size_t i = 0; i < n; i++)
//         memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

//     fclose(f);
//     return x;
// }

// // not very clean, but works as long as sizeof(int) == sizeof(float)
// int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
//     return (int*)fvecs_read(fname, d_out, n_out);
// }

// double elapsed() {
//     struct timeval tv;
//     gettimeofday(&tv, nullptr);
//     return tv.tv_sec + tv.tv_usec * 1e-6;
// }
// int main() {

//      double t0 = elapsed();

//     size_t dimensions, num_training_vectors;
//     size_t num_base_vectors;
//     float* training_set = fvecs_read("sift/sift_learn.fvecs", &dimensions, &num_training_vectors);
//     float* base_set = fvecs_read("sift/sift_base.fvecs", &dimensions, &num_base_vectors);
//     printf("[%.3f s]\n", elapsed()-t0);
//     // printing data
//     for(size_t i=0;i<num_training_vectors;i++){
//         for(size_t j=0;j<dimensions;j++){
//             std::cout << std::fixed << std::setprecision(2) << *(training_set+i+j)<<" ";
//         }
//         std::cout<<"\n";
//     }

//     std::cout<<dimensions<<"\n";
//     std::cout<<num_training_vectors<<"\n";

//     faiss::IndexFlatL2 coarse_quantizer(dimensions);

//     size_t num_centroids = 16;

//     faiss::IndexIVFFlat index(&coarse_quantizer, dimensions, num_centroids);
//     index.verbose = true;
//     index.train(num_training_vectors, training_set);

//     printf("[%.3f s]\n", elapsed()-t0);
//     float* centroids0 = new float[num_centroids*dimensions];
//     std::cout<<"quantizer ntotal before adding: "<<index.quantizer->ntotal<<"\n";
//     index.quantizer->reconstruct_n(0, num_centroids, centroids0);
//     std::cout<<"-----centroids0 start-----\n";
//     for(size_t i=0;i<num_centroids;i++){
//         for(size_t j=0;j<dimensions;j++){
//              std::cout << std::fixed << std::setprecision(2) << *(centroids0+i+j)<<" ";
//             // std::cout<<int(centroids[i+j])<<" ";
//         }
//         std::cout<<"\n";
//     }
//     std::cout<<"-----centroids0 end-----\n";

//     index.add(num_base_vectors, base_set);

//     std::cout<<"quantizer ntotal after adding: "<<index.quantizer->ntotal<<"\n";
//     float* centroids1 = new float[num_centroids*dimensions];

//     index.quantizer->reconstruct_n(0, num_centroids, centroids1);


//     std::cout<<"-----centroids1 start-----\n";
//     for(size_t i=0;i<num_centroids;i++){
//         for(size_t j=0;j<dimensions;j++){
//              std::cout << std::fixed << std::setprecision(2) << *(centroids1+i+j)<<" ";
//             // std::cout<<int(centroids[i+j])<<" ";
//         }
//         std::cout<<"\n";
//     }
//     std::cout<<"-----centroids1 end-----\n";
//     // std::vector<float> centroids = index.quantizer->centroids;
//     // std::cout<<"-----centroids start-----\n";
//     // for(size_t i=0;i<num_centroids;i++){
//     //     for(size_t j=0;j<dimensions;j++){
//     //          std::cout << std::fixed << std::setprecision(2) << centroids[i+j]<<" ";
//     //         // std::cout<<int(centroids[i+j])<<" ";
//     //     }
//     //     std::cout<<"\n";
//     // }
//     // std::cout<<"-----centroids end-----\n";

//     // float* centroids2 = new float[num_centroids * dimensions];
//     // faiss::kmeans_clustering(dimensions, num_training_vectors, num_centroids, training_set, centroids2);

//     // std::cout<<"-----centroids2 start-----\n";
//     // for(size_t i=0;i<num_centroids;i++){
//     //     for(size_t j=0;j<dimensions;j++){
//     //         std::cout << std::fixed << std::setprecision(2) << *(centroids2+i+j)<<" ";
//     //         // std::cout<<int(*(centroids+i+j))<<" ";
//     //     }
//     //     std::cout<<"\n";
//     // }
//     // std::cout<<"-----centroids2 end-----\n";


//     // // faiss::IndexHNSWFlat hnsw_index(dimensions, num_centroids, faiss::METRIC_L2);
//     // // hnsw_index.train(num_centroids, centroids2);



//     // std::string index_factory_string = "IVF" + std::to_string(num_centroids) + ",Flat";
//     // faiss::index_factory_verbose = 1;
//     // faiss::Index* index2 = faiss::index_factory(dimensions, index_factory_string.c_str());
//     // // index2->train(num_training_vectors, training_set);
//     // index2->is_trained = true;
//     // index2->add(num_training_vectors, training_set);
//     // std::cout<<index2->ntotal<<"\n";
//     // float* centroids3 = new float[num_centroids*dimensions];
//     // index2->reconstruct_n(0, num_centroids,centroids3);

//     // std::cout<<"-----centroids3 start-----\n";
//     // for(size_t i=0;i<num_centroids;i++){
//     //     for(size_t j=0;j<dimensions;j++){
//     //          std::cout << std::fixed << std::setprecision(2) << *(centroids3+i+j)<<" ";
//     //     }
//     //     std::cout<<"\n";
//     // }
//     // std::cout<<"-----centroids3 end-----\n";


// }

// #include <faiss/index_io.h>
// #include <faiss/index_factory.h>

// int main() {
//     // Initialize Faiss and your dataset (xb, xq, and other variables)

//     int d = 128; // Dimension of your data
//     int nlist = 100; // Number of clusters for IVF
//     int nprobe = 32; // Number of probes (vary this as needed)

//     // Define the parameters string with nprobe
//     std::string ivf_params = "Flat,IVF" + faiss::index_key_str("Flat") + ",nlist=" + std::to_string(nlist) +
//                             ",nprobe=" + std::to_string(nprobe);

//     // Create the index using index_factory with the parameters string
//     faiss::Index* index = faiss::index_factory(d, ivf_params);

//     // Train or add data to the index
//     index->train(n, xb);  // Train the index with training data
//     index->add(n, xb);    // Add the data to the index

//     // Perform a search with the specified nprobe
//     int k = 10; // Number of nearest neighbors to retrieve
//     index->search(nq, xq, k, distances, indices);

//     // Save the index to disk (if needed)
//     faiss::write_index(index, "your_index_file.index");

//     // Clean up resources
//     delete index;

//     return 0;
// }

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
int main(int argc, char* argv[]) {
    assert(sizeof(int) == sizeof(float));
    auto start = timeNow(), end = timeNow();
    faiss::Index* index;

    std::string use_pre_existing = argv[1];

    if (use_pre_existing == "1") {
        index = faiss::read_index(argv[5]);
        index->verbose = true;

    } else {
        size_t training_set_dimensions, num_training_vectors;
        start = timeNow();
        float* training_set = fvecs_read(
                argv[3], &training_set_dimensions, &num_training_vectors);
        end = timeNow();
        printf("Time taken to read training set: %.4fs\n", elapsed(start, end));

        std::string index_factory_string = argv[2];
         int d = 128; // Dimension of your data
        int nlist = 100; // Number of clusters for IVF
        int nprobe = 32; // Number of probes (vary this as needed)

        // Define the parameters string with nprobe
    
        std::string ivf_params = "Flat,IVFFlat,nlist=" + std::to_string(nlist) +
                                ",nprobe=" + std::to_string(nprobe);
        index = faiss::index_factory(
                training_set_dimensions, ivf_params.c_str());
        index->verbose = true;

        start = timeNow();
        index->train(num_training_vectors, training_set);
        end = timeNow();
        printf("Time taken to train index: %.4fs\n", elapsed(start, end));

        delete[] training_set;

        std::string base_set = argv[4];
        if (base_set!="nil"){
            size_t base_set_dimensons, num_base_vectors;

            start = timeNow();
            float* base_set =
                    fvecs_read(argv[4], &base_set_dimensons, &num_base_vectors);
            end = timeNow();
            assert(training_set_dimensions == base_set_dimensons);
            printf("Time taken to read training set: %.4fs\n", elapsed(start, end));

            start = timeNow();
            index->add(num_base_vectors, base_set);
            end = timeNow();
            printf("Time taken to add base vectors to the index: %.4fs\n",
                elapsed(start, end));
            delete[] base_set;
        } else {
            printf("base set is nil, not adding vectors to the index\n");
        }

        faiss::write_index(index, argv[5]);
    }

    size_t query_set_dimensions, num_queries;
    start = timeNow();
    float* query_set = fvecs_read(argv[6], &query_set_dimensions, &num_queries);
    end = timeNow();
    printf("Time taken to read query set: %.4fs\n", elapsed(start, end));
    size_t num_groundtruth, groundtruth_dimensions;
    start = timeNow();
    int* gt_int =
            ivecs_read(argv[7], &groundtruth_dimensions, &num_groundtruth);
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

    start = timeNow();
    // printf("index nprobe: %d\n", ParameterS);
    index->search(
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