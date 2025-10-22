#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define EPSILON 1e-10
#define MAX_ITERATIONS 1000

typedef struct {
    double **data;
    int rows;
    int cols;
} Matrix;

Matrix* create_matrix(int rows, int cols) {
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m->data[i] = calloc(cols, sizeof(double));
    }
    return m;
}

void free_matrix(Matrix *m) {
    if (m) {
        for (int i = 0; i < m->rows; i++) {
            free(m->data[i]);
        }
        free(m->data);
        free(m);
    }
}

void print_matrix(Matrix *m, const char *name) {
    printf("\n%s (%dx%d):\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f ", m->data[i][j]);
        }
        printf("\n");
    }
}

Matrix* multiply_matrices(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Error: Dimensiones incompatibles para multiplicación\n");
        return NULL;
    }
    
    Matrix *C = create_matrix(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            for (int k = 0; k < A->cols; k++) {
                C->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    return C;
}

Matrix* transpose(Matrix *A) {
    Matrix *T = create_matrix(A->cols, A->rows);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j][i] = A->data[i][j];
        }
    }
    return T;
}

Matrix* read_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: No se puede abrir el archivo %s\n", filename);
        return NULL;
    }
    
    char line[MAX_LINE_LENGTH];
    int rows = 0, cols = 0;
    

    while (fgets(line, sizeof(line), file)) {
        if (rows == 0) {
            char *token = strtok(line, ",");
            while (token) {
                cols++;
                token = strtok(NULL, ",");
            }
        }
        rows++;
    }
    
    printf("Dimensiones detectadas: %d filas, %d columnas\n", rows, cols);
    
    Matrix *data = create_matrix(rows, cols);
    rewind(file);
    
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < rows) {
        char *token = strtok(line, ",");
        int col = 0;
        while (token && col < cols) {
            data->data[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    
    fclose(file);
    return data;
}

void write_csv(Matrix *data, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: No se puede crear el archivo %s\n", filename);
        return;
    }
    
    for (int i = 0; i < data->rows; i++) {
        for (int j = 0; j < data->cols; j++) {
            fprintf(file, "%.6f", data->data[i][j]);
            if (j < data->cols - 1) fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Datos guardados en %s\n", filename);
}

double* calculate_mean(Matrix *data) {
    double *mean = calloc(data->cols, sizeof(double));
    for (int j = 0; j < data->cols; j++) {
        for (int i = 0; i < data->rows; i++) {
            mean[j] += data->data[i][j];
        }
        mean[j] /= data->rows;
    }
    return mean;
}

Matrix* center_data(Matrix *data) {
    double *mean = calculate_mean(data);
    Matrix *centered = create_matrix(data->rows, data->cols);
    
    for (int i = 0; i < data->rows; i++) {
        for (int j = 0; j < data->cols; j++) {
            centered->data[i][j] = data->data[i][j] - mean[j];
        }
    }
    
    free(mean);
    return centered;
}

Matrix* calculate_covariance(Matrix *centered_data) {
    Matrix *transposed = transpose(centered_data);
    Matrix *cov = multiply_matrices(transposed, centered_data);
    
    double factor = 1.0 / (centered_data->rows - 1);
    for (int i = 0; i < cov->rows; i++) {
        for (int j = 0; j < cov->cols; j++) {
            cov->data[i][j] *= factor;
        }
    }
    
    free_matrix(transposed);
    return cov;
}

double power_method(Matrix *A, double *eigenvector, int max_iter) {
    int n = A->rows;
    double *v = malloc(n * sizeof(double));
    double *v_new = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        v[i] = 1.0;
    }
    
    double eigenvalue = 0;
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < n; i++) {
            v_new[i] = 0;
            for (int j = 0; j < n; j++) {
                v_new[i] += A->data[i][j] * v[j];
            }
        }
        
        double numerator = 0, denominator = 0;
        for (int i = 0; i < n; i++) {
            numerator += v[i] * v_new[i];
            denominator += v[i] * v[i];
        }
        eigenvalue = numerator / denominator;
        
        double norm = 0;
        for (int i = 0; i < n; i++) {
            norm += v_new[i] * v_new[i];
        }
        norm = sqrt(norm);
        
        for (int i = 0; i < n; i++) {
            v_new[i] /= norm;
        }
        
        double diff = 0;
        for (int i = 0; i < n; i++) {
            diff += fabs(v_new[i] - v[i]);
        }
        
        for (int i = 0; i < n; i++) {
            v[i] = v_new[i];
        }
        
        if (diff < EPSILON) break;
    }
    
    for (int i = 0; i < n; i++) {
        eigenvector[i] = v[i];
    }
    
    free(v);
    free(v_new);
    return eigenvalue;
}

void deflate_matrix(Matrix *A, double eigenvalue, double *eigenvector) {
    int n = A->rows;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A->data[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
}

Matrix* pca_transform(Matrix *data, int k) {
    printf("Datos originales: %d muestras, %d dimensiones\n", data->rows, data->cols);
    printf("Reduciendo a %d dimensiones\n", k);
    
    printf("1. Centrando los datos...\n");
    Matrix *centered = center_data(data);

    printf("2. Calculando matriz de covarianza...\n");
    Matrix *cov = calculate_covariance(centered);
    printf("Matriz de covarianza: %dx%d\n", cov->rows, cov->cols);
    
    printf("3. Encontrando %d componentes principales...\n", k);
    Matrix *components = create_matrix(k, data->cols);
    double *eigenvalues = malloc(k * sizeof(double));
    
    Matrix *cov_copy = create_matrix(cov->rows, cov->cols);
    for (int i = 0; i < cov->rows; i++) {
        for (int j = 0; j < cov->cols; j++) {
            cov_copy->data[i][j] = cov->data[i][j];
        }
    }
    
    for (int i = 0; i < k; i++) {
        double *eigenvector = malloc(data->cols * sizeof(double));
        eigenvalues[i] = power_method(cov_copy, eigenvector, MAX_ITERATIONS);
        
        printf("   Componente %d: eigenvalor = %.6f\n", i+1, eigenvalues[i]);
        
        for (int j = 0; j < data->cols; j++) {
            components->data[i][j] = eigenvector[j];
        }
        
        if (i < k - 1) {
            deflate_matrix(cov_copy, eigenvalues[i], eigenvector);
        }
        
        free(eigenvector);
    }
    
    printf("4. Proyectando datos...\n");
    Matrix *projected = multiply_matrices(centered, transpose(components));
    
    printf("Datos proyectados: %dx%d\n", projected->rows, projected->cols);
    
    free_matrix(centered);
    free_matrix(cov);
    free_matrix(cov_copy);
    free_matrix(components);
    free(eigenvalues);
    
    return projected;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <archivo_entrada.csv> <archivo_salida.csv> <num_componentes>\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int k = atoi(argv[3]);
    
    printf("algoritmo PCA en C ===\n");
    printf("Archivo de entrada: %s\n", input_file);
    printf("Archivo de salida: %s\n", output_file);
    printf("Número de componentes: %d\n", k);
    
    Matrix *data = read_csv(input_file);
    if (!data) {
        return 1;
    }
    
    if (k > data->cols) {
        printf("Error: k (%d) no puede ser mayor que el número de dimensiones (%d)\n", k, data->cols);
        free_matrix(data);
        return 1;
    }
    
    Matrix *projected = pca_transform(data, k);
    
    write_csv(projected, output_file);
    
    free_matrix(data);
    free_matrix(projected);
    
    return 0;

}


