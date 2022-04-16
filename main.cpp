#include <opencv2/opencv.hpp>
#include <cmath>
#include <tbb/parallel_for.h>
#include <immintrin.h>
#include "avx_mathfun.h"

using namespace cv;
using namespace std;

class WeightFunctions
{
public:
    inline static float g1(float value, float k_l)
    {
        return expf(-value * value / (k_l * k_l));
    }

    inline static __m256 g1(__m256 value, float k_l)
    {
        float ik2 = -1.0f / (k_l * k_l);
        return exp256_ps(_mm256_mul_ps(_mm256_mul_ps(value, value), _mm256_broadcast_ss(&ik2)));
    }

    inline static __m256 g1(__m256 value, __m256 ik)
    {
        return exp256_ps(_mm256_mul_ps(_mm256_mul_ps(value, value), ik));
    }

    inline static float g2(float value, float k_l)
    {
        return 1.0f / (1.0f + value * value / (k_l * k_l));
    }

    inline static __m256 g2(__m256 value, float k_l)
    {
        float ik2 = 1.0f / (k_l * k_l);
        static const float one = 1.0f;
        return _mm256_rcp_ps(
                _mm256_add_ps(
                        _mm256_broadcast_ss(&one),
                        _mm256_mul_ps(_mm256_mul_ps(value, value),_mm256_broadcast_ss(&ik2))));
    }

    inline static __m256 g2(__m256 value, __m256 ik)
    {
        static const float one = 1.0f;
        return _mm256_rcp_ps(
                _mm256_add_ps(
                        _mm256_broadcast_ss(&one),
                        _mm256_mul_ps(_mm256_mul_ps(value, value), ik)));
    }

};

class Original_AD
{
private:
    float lambda;
    float k;
    int n_iters;
    int func;

private:
    float g(float value, float k_l) const
    {
        switch (func)
        {
            case 1:
                return WeightFunctions::g1(value, k_l);
                
            case 2:
                return WeightFunctions::g2(value, k_l);
                
            default:
                return 0.0f;
        }
    } 
    
public:
    Original_AD (float lambda, float k, int n_iters, int func)
        : lambda(lambda), k(k), n_iters(n_iters), func(func) {}

public:
    Mat anisotropic_diffusion(Mat img)
    {
        img.convertTo(img, CV_32FC1, 1.0f / 255.0f);
        Mat Ipred(img.rows + 1, img.cols + 1, CV_32FC1);
        copyMakeBorder(img, Ipred, 1, 1, 1, 1, BORDER_CONSTANT);
        Mat Inext = Ipred.clone();
        for (int n = 0; n < n_iters; n++)
        {
            tbb::parallel_for(1, img.rows - 1, [&](int i)
            {
                for (int j = 1; j < img.cols - 1; j++)
                {
                    float sum = 0;

                    sum += g(abs(Ipred.at<float>(i + 1, j) - Ipred.at<float>(i, j)), k) *
                           Ipred.at<float>(i + 1, j) - Ipred.at<float>(i, j);

                    sum += g(abs(Ipred.at<float>(i - 1, j) - Ipred.at<float>(i, j)), k) *
                           Ipred.at<float>(i - 1, j) - Ipred.at<float>(i, j);

                    sum += g(abs(Ipred.at<float>(i, j + 1) - Ipred.at<float>(i, j)), k) *
                           Ipred.at<float>(i, j + 1) - Ipred.at<float>(i, j);

                    sum += g(abs(Ipred.at<float>(i, j - 1) - Ipred.at<float>(i, j)), k) *
                           Ipred.at<float>(i, j - 1) - Ipred.at<float>(i, j);
                    sum = lambda * sum;

                    Inext.at<float>(i, j) = Ipred.at<float>(i, j) + sum;
                }
            });
            Ipred.release();
            Ipred = Inext.clone();
        }
        return Ipred.clone();
    }
};

class Patch_based_AD
{
private :
    int r;
    int n_iters;
    int func;
    int k;
    float kr;
    float kr2, ikr2;
    float ks;
    float ks2, iks2;
    float lambda;
    float sigma;

public:
    Patch_based_AD(int r, int n_iters, int func, int k, float kr, float ks, float lambda, float sigma)
        : r(r), n_iters(n_iters), func(func), k(k), kr(kr), ks(ks), lambda(lambda), sigma(sigma)
    {
        kr2 = 1.0f / (kr * kr);
        ikr2 = -1.0f / (kr * kr);
        ks2 = 1.0f / (ks * ks);
        iks2 = -1.0f / (ks * ks);
    }

private:
    float l(const Mat& img) const
    {
        Mat img_N = img(Rect(2 * r + 1, 2 * r, img.cols - 2 * (r + 1), img.rows - 2 * (r + 1))).clone();
        Mat img_S = img(Rect(2 * r + 1, 2 * r + 2, img.cols - 2 * (r + 1), img.rows - 2 * (r + 1))).clone();
        Mat img_W = img(Rect(2 * r, 2 * r + 1, img.cols - 2 * (r + 1), img.rows - 2 * (r + 1))).clone();
        Mat img_E = img(Rect(2 * r + 2, 2 * r + 1, img.cols - 2 * (r + 1), img.rows - 2 * (r + 1))).clone();

        Mat sum1;
        cv::absdiff(img_N, img_S, sum1);
        cv::pow(sum1, 2, sum1);

        Mat sum2;
        cv::absdiff(img_W, img_E, sum2);
        cv::pow(sum2, 2, sum2);

        Mat sum3 = sum1 + sum2;
        cv::sqrt(sum3, sum3);

        float sum = cv::sum(sum3).val[0];
        sum = sum / 2;
        return lambda * (1.0 - float(exp(-float((pow(sum, 2))) / float(pow((img.rows - 2 * r)
                                                                              * (img.cols - 2 * r) * sigma, 2)))));
    }

private:
    float G1(const float *p, const float *q, size_t stride) const
    {
        float znam = 0.0f;
        float sum = 0.0f;

        for (int ip = -r; ip <= r; ip++)
            for (int jp = -r; jp <= r; jp++)
            {
                float d = expf((float)(ip * ip + jp * jp) * iks2);
                znam += d;

                auto v = p[ip * stride + jp];
                auto s0 = 0.0f;

                for (int iq = -r; iq <= r; iq++)
                    for (int jq = -r; jq <= r; jq++)
                    {
                        auto w = q[ip * stride + jp] - v;
                        s0 += expf(w * w * ikr2);
                    }

                sum += s0 * d;
            }

        return sum / znam;
    }

    __m256 G1V(const float *p, const float *q, size_t stride) const
    {
        __m256 znam =_mm256_setzero_ps();
        __m256 sum =_mm256_setzero_ps();

        for (int ip = -r; ip <= r; ip++)
            for (int jp = -r; jp <= r; jp++)
            {
                float d = expf((float)(ip * ip + jp * jp) * iks2);
                znam += d;

                __m256 v = _mm256_loadu_ps(p + ip * stride + jp);
                __m256 s0 = _mm256_setzero_ps();

                for (int iq = -r; iq <= r; iq++)
                    for (int jq = -r; jq <= r; jq++)
                    {
                        __m256 w = _mm256_sub_ps(_mm256_loadu_ps(q + ip * stride + jp), v); // w = img - v
                        s0 = _mm256_add_ps(s0, WeightFunctions::g1(w, _mm256_broadcast_ss(&ikr2)));
                    }

                sum = _mm256_add_ps(sum, _mm256_mul_ps(s0, _mm256_broadcast_ss(&d)));
            }

        return _mm256_div_ps(sum, znam);
    }

    float G2(const float *p, const float *q, size_t stride) const
    {
        float znam = 0.0f;
        float sum = 0.0f;

        for (int ip = -r; ip <= r; ip++)
            for (int jp = -r; jp <= r; jp++)
            {
                float d = 1.0f / (1.0f + (float)(ip * ip + jp * jp) * ks2);
                znam += d;

                auto v = p[ip * stride + jp];
                auto s0 = 0.0f;

                for (int iq = -r; iq <= r; iq++)
                    for (int jq = -r; jq <= r; jq++)
                    {
                        auto w = q[ip * stride + jp] - v;
                        s0 += 1.0f / (1.0f + w * w / (kr * kr));
                    }

                sum += s0 * d;
            }

        return sum / znam;
    }

    __m256 G2V(const float *p, const float *q, size_t stride) const
    {
        static const float one = 1.0f;

        __m256 znam =_mm256_setzero_ps();
        __m256 sum =_mm256_setzero_ps();

        for (int ip = -r; ip <= r; ip++)
            for (int jp = -r; jp <= r; jp++)
            {
                float d = 1.0f / (1.0f + (float)(ip * ip + jp * jp) * ks2);
                znam += d;

                __m256 v = _mm256_loadu_ps(p + ip * stride + jp);
                __m256 s0 = _mm256_setzero_ps();

                for (int iq = -r; iq <= r; iq++)
                    for (int jq = -r; jq <= r; jq++)
                    {
                        __m256 w = _mm256_sub_ps(_mm256_loadu_ps(q + ip * stride + jp), v); // w = img - v
                        s0 = _mm256_add_ps(s0, WeightFunctions::g2(w, _mm256_broadcast_ss(&kr2)));
                    }

                sum = _mm256_add_ps(sum, _mm256_mul_ps(s0, _mm256_broadcast_ss(&d)));
            }

        return _mm256_div_ps(sum, znam);
    }

    void ProcessPixel(const float *prev, float *dst, size_t stride, float lt) const
    {
        float sum_G = 0;
        float sum_g = 0;
        float H_g = 0;
        float H_G = 0;

        for (int ii = -1; ii <= 1; ii++)
            for (int jj = -1; jj <= 1; jj++)
            {
                const float *prev1 = prev + ii * stride + jj;

                float patch = func == 1 ? G1(prev, prev1, stride) : G2(prev, prev1, stride);
                float dop = *prev1 - *prev;
                float local = func == 1 ? WeightFunctions::g1(abs(dop), k) : WeightFunctions::g2(abs(dop), k);

                H_G += patch;
                sum_G += patch * dop;

                H_g += local;
                sum_g += local * dop;
            }

        H_G = float(H_G) / float(8);
        H_g = float(H_g) / float(8);
        float new_dop = lt * (H_G * sum_G + H_g * sum_g);
        float new_val = *prev + new_dop;
        *dst = new_val;
    }

    void ProcessPixelV(const float *prev, float *dst, size_t stride, float lt) const
    {
        const static float f8 = 8.0f;

        __m256 sum_G = _mm256_setzero_ps();
        __m256 sum_g = _mm256_setzero_ps();
        __m256 H_g = _mm256_setzero_ps();
        __m256 H_G = _mm256_setzero_ps();

        for (int ii = -1; ii <= 1; ii++)
            for (int jj = -1; jj <= 1; jj++)
            {
                const float *prev1 = prev + ii * stride + jj;

                __m256 patch = func == 1 ? G1V(prev, prev1, stride) : G2V(prev, prev1, stride);
                __m256 dop = _mm256_sub_ps(_mm256_loadu_ps(prev1), _mm256_loadu_ps(prev));
                __m256 local = func == 1 ? WeightFunctions::g1(dop, k) : WeightFunctions::g2(dop, k);

                H_G = _mm256_add_ps(H_G, patch);
                sum_G = _mm256_add_ps(sum_G, _mm256_mul_ps(patch, dop));

                H_g = _mm256_add_ps(H_g, local);
                sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(local, dop));
            }

        H_G = _mm256_div_ps(H_G, _mm256_broadcast_ss(&f8));
        H_g = _mm256_div_ps(H_g, _mm256_broadcast_ss(&f8));

        __m256 new_dop = _mm256_mul_ps(_mm256_broadcast_ss(&lt),
                                       _mm256_add_ps(
                                               _mm256_mul_ps(H_G, sum_G),
                                               _mm256_mul_ps(H_g, sum_g)));

        _mm256_storeu_ps(dst, _mm256_add_ps(_mm256_loadu_ps(prev), new_dop));
    }

public:
    Mat anisotropic_diffusion(Mat img)
    {
        img.convertTo(img, CV_32FC1, 1.f/255);
        Mat Ipred(img.rows + 2 * (r + 1), img.cols + 2 * (r + 1), CV_32FC1);

        copyMakeBorder(img, Ipred, r + 1, r + 1, r + 1, r + 1, BORDER_CONSTANT);

        Mat Inext = Ipred.clone();

        for (int n = 0; n < n_iters; n++)
        {
            float lt = l(Ipred);

            tbb::parallel_for(r + 1, img.rows - r - 1, [&](int i)
            {
                _mm256_zeroupper();

                int j1 = img.cols - 2 * r - 2;
                int j8 = j1 / 8 * 8;

                for (int j = 0; j < j8; j += 8)
                {
                    ProcessPixelV(Ipred.ptr<float>(i, j + r + 1), Inext.ptr<float>(i, j + r + 1), Ipred.step / sizeof(float), lt);
                }

                for (int j = j8; j < j1; j++)
                {
                    ProcessPixel(Ipred.ptr<float>(i, j + r + 1), Inext.ptr<float>(i, j + r + 1), Ipred.step / sizeof(float), lt);
                }
            });
            Inext.copyTo(Ipred);
        }
        return Ipred;
    }
};

int main()
{
    // Mat img = imread("/home/andrew/tmp/i02_01_5.bmp", IMREAD_GRAYSCALE);
    Mat img = imread("/home/andrew/University/Data/Std/Lena.bmp", IMREAD_GRAYSCALE);

    float l = 1.0f / 4.0f;
    float l1 = 1.0f / 16.0f;

    auto start_ad = clock();
    Original_AD first(l, 100, 10, 1);
    Mat img1 = first.anisotropic_diffusion(img);
    auto end_ad = clock();

    auto start_ad1 = clock();
    Patch_based_AD second(2, 100, 1, 10, 10, 10, l1, 1);
    Mat img2 = second.anisotropic_diffusion(img);
    auto end_ad1 = clock();

    cout << "AD PM image time \n";
    cout << (float)(end_ad - start_ad) / CLOCKS_PER_SEC << endl;

    cout << "Patch-based image time \n";
    cout << (float)(end_ad1 - start_ad1) / CLOCKS_PER_SEC << endl;

    imshow("Patch-based image", img2);
    imshow("AD PM image", img1);
    imshow("Original image", img);
    waitKey(0);
    return 0;
}
