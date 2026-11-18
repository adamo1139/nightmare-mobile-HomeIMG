#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Independent HomeIMG CPU interface. Buffers are caller-owned. Callbacks run synchronously.
// flags: 1 = CPU txt2img; no HTP capability is advertised by this implementation.
uint32_t homeimg_api_version(void);
uint32_t homeimg_capabilities(void);
typedef int (*homeimg_progress)(void *user, int completed, int total);
// Optional dense-projection backend. All arrays are contiguous FP32 row-major:
 // x[tokens,in], weight[out,in], result[tokens,out]. Return 0 to recompute on CPU.
typedef int (*homeimg_projection)(void *user,const float *x,const float *weight,
                                  int tokens,int in,int out,float *result);
typedef int (*homeimg_attention)(void *user,const float *q,const float *k,const float *v,
                                 int tokens,int width,float scale,float *output);
typedef int (*homeimg_masked_attention)(void*,const float*,const float*,const float*,int,int,int,float,float*,int);
enum homeimg_row_kind { HOMEIMG_RMS, HOMEIMG_RESIDUAL, HOMEIMG_EULER, HOMEIMG_BIAS, HOMEIMG_SILU, HOMEIMG_SWIGLU, HOMEIMG_ROUTES, HOMEIMG_TEXT_ROPE };
typedef struct {
    enum homeimg_row_kind kind;
    int rows,cols;
    const float *x,*extra,*negative,*norm,*gain,*bias;
    float dt,cfg;
    float *out;
} homeimg_rows;
typedef int (*homeimg_rowop)(void*,const homeimg_rows*);
typedef int (*homeimg_spatial)(void*,const float*,int,int,int,float,const float*,const float*,float*);
typedef int (*homeimg_conv)(void*,const float*,const float*,int,int,int,int,float*);
typedef int (*homeimg_group)(void*,const float*,const float*,const float*,int,int,int,float*);
typedef int (*homeimg_expert)(void*,const float*,const int8_t*,const float*,int,int,int,float*);
typedef int (*homeimg_mlp)(void*,float*,int,int,int,const float*,const float*,const float*,const float*,const float*,const float*);
typedef int (*homeimg_cross)(void*,const float*,const float*,const float*,int,int,int,float,float*);
// Input: FP32 [side*side,128], token-major. Output: packed RGB8 [16*side,16*side,3].
// root points to a compatible HomeIMG checkpoint directory. side ranges from 16 to 64 inclusive.
// Returns 0 on success, 1 on error (message in error), 2 on callback cancellation.
int homeimg_generate(const char *root,const char *prompt,const char *negative,
                     int side,int steps,float cfg,const float *noise,size_t noise_count,
                     uint8_t *rgb,size_t rgb_bytes,homeimg_progress progress,void *user,
                     char *error,size_t error_bytes);
int homeimg_generate_projected(const char *root,const char *prompt,const char *negative,
                     int side,int steps,float cfg,const float *noise,size_t noise_count,
                     uint8_t *rgb,size_t rgb_bytes,homeimg_progress progress,void *user,
                     char *error,size_t error_bytes,homeimg_projection project,void *backend);
// time_shift is zero for checkpoint defaults, otherwise finite and positive.
int homeimg_generate_shifted(const char *root,const char *prompt,const char *negative,
                     int side,int steps,float cfg,const float *noise,size_t noise_count,
                     uint8_t *rgb,size_t rgb_bytes,homeimg_progress progress,void *user,
                     char *error,size_t error_bytes,homeimg_projection project,void *backend,float time_shift);
int homeimg_generate_accelerated(const char *root,const char *prompt,const char *negative,
                     int side,int steps,float cfg,const float *noise,size_t noise_count,
                     uint8_t *rgb,size_t rgb_bytes,homeimg_progress progress,void *user,
                     char *error,size_t error_bytes,homeimg_projection project,void *backend,float time_shift,
                     homeimg_attention attention);
// Optional Vulkan operations. Their context is shared with the projection callback.
typedef void (*homeimg_weight_scope)(void*,const void*,size_t);
typedef struct {
    homeimg_attention attention;
    homeimg_mlp mlp;
    homeimg_cross cross;
    homeimg_conv conv;
    homeimg_masked_attention masked;
    homeimg_spatial spatial;
    homeimg_rowop rowop;
    homeimg_expert expert;
    homeimg_group group;
    homeimg_weight_scope weight_scope;
} homeimg_device_ops;
int homeimg_generate_blocks(const char *root,const char *prompt,const char *negative,
                     int side,int steps,float cfg,const float *noise,size_t noise_count,
                     uint8_t *rgb,size_t rgb_bytes,homeimg_progress progress,void *user,
                     char *error,size_t error_bytes,homeimg_projection project,void *backend,float time_shift,
                     const homeimg_device_ops *ops);
#ifdef __cplusplus
}
#endif
