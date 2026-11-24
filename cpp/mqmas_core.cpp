#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mqmas {

using cd = std::complex<double>;
constexpr cd Iu{0.0, 1.0};

template<int D> using Vec = std::array<cd, D>;
template<int D> using Mat = std::array<cd, D * D>;

template<int D> constexpr int mi(int r, int c) noexcept { return r * D + c; }

template<int D>
std::array<double, D> m_values() {
    const double spin = 0.5 * static_cast<double>(D - 1);
    std::array<double, D> m{};
    for (int i = 0; i < D; ++i) m[i] = spin - static_cast<double>(i);
    return m;
}

template<int D>
Mat<D> ix_matrix() {
    Mat<D> out{};
    const double spin = 0.5 * static_cast<double>(D - 1);
    const auto m = m_values<D>();
    for (int i = 0; i < D - 1; ++i) {
        const double v = 0.5 * std::sqrt(spin * (spin + 1.0) - m[i] * (m[i] - 1.0));
        out[mi<D>(i, i + 1)] = v;
        out[mi<D>(i + 1, i)] = v;
    }
    return out;
}

template<int D>
Mat<D> iy_matrix() {
    Mat<D> out{};
    const double spin = 0.5 * static_cast<double>(D - 1);
    const auto m = m_values<D>();
    for (int i = 0; i < D - 1; ++i) {
        const double v = 0.5 * std::sqrt(spin * (spin + 1.0) - m[i] * (m[i] - 1.0));
        out[mi<D>(i, i + 1)] = -Iu * v;
        out[mi<D>(i + 1, i)] = Iu * v;
    }
    return out;
}

template<int D>
struct RealEigen {
    std::array<double, D> val{};
    std::array<double, D * D> vec{}; // real, columns
};

template<int D>
RealEigen<D> jacobi_real_symmetric(std::array<double, D * D> a) {
    RealEigen<D> e;
    for (int i = 0; i < D; ++i) e.vec[i * D + i] = 1.0;
    for (int iter = 0; iter < 200 * D * D; ++iter) {
        int p = 0, q = 1;
        double vmax = 0.0;
        for (int i = 0; i < D; ++i) {
            for (int j = i + 1; j < D; ++j) {
                const double v = std::abs(a[i * D + j]);
                if (v > vmax) { vmax = v; p = i; q = j; }
            }
        }
        if (vmax < 1e-15) break;
        const double app = a[p * D + p];
        const double aqq = a[q * D + q];
        const double apq = a[p * D + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        for (int k = 0; k < D; ++k) {
            if (k == p || k == q) continue;
            const double akp = a[k * D + p];
            const double akq = a[k * D + q];
            const double nkp = c * akp - s * akq;
            const double nkq = s * akp + c * akq;
            a[k * D + p] = a[p * D + k] = nkp;
            a[k * D + q] = a[q * D + k] = nkq;
        }
        a[p * D + p] = c*c*app - 2*s*c*apq + s*s*aqq;
        a[q * D + q] = s*s*app + 2*s*c*apq + c*c*aqq;
        a[p * D + q] = a[q * D + p] = 0.0;
        for (int k = 0; k < D; ++k) {
            const double vkp = e.vec[k * D + p];
            const double vkq = e.vec[k * D + q];
            e.vec[k * D + p] = c * vkp - s * vkq;
            e.vec[k * D + q] = s * vkp + c * vkq;
        }
    }
    for (int i = 0; i < D; ++i) e.val[i] = a[i * D + i];
    std::array<int, D> order{};
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int aidx, int bidx){ return e.val[aidx] < e.val[bidx]; });
    RealEigen<D> out;
    for (int j = 0; j < D; ++j) {
        const int old = order[j];
        out.val[j] = e.val[old];
        for (int i = 0; i < D; ++i) out.vec[i * D + j] = e.vec[i * D + old];
    }
    return out;
}

template<int D>
inline Vec<D> mat_vec(const Mat<D>& a, const Vec<D>& x) {
    Vec<D> y{};
    for (int i = 0; i < D; ++i) {
        cd s{0.0, 0.0};
        for (int j = 0; j < D; ++j) s += a[mi<D>(i,j)] * x[j];
        y[i] = s;
    }
    return y;
}

template<int D>
inline Vec<D> mat_dag_vec(const Mat<D>& a, const Vec<D>& x) {
    Vec<D> y{};
    for (int i = 0; i < D; ++i) {
        cd s{0.0, 0.0};
        for (int j = 0; j < D; ++j) s += std::conj(a[mi<D>(j,i)]) * x[j];
        y[i] = s;
    }
    return y;
}

template<int D>
inline cd inner(const Vec<D>& x, const Vec<D>& y) {
    cd s{0.0,0.0};
    for (int i = 0; i < D; ++i) s += std::conj(x[i]) * y[i];
    return s;
}

template<int D>
inline Vec<D> apply_diag(const cd* e, const Vec<D>& x, bool dagger=false) {
    Vec<D> y{};
    for (int i = 0; i < D; ++i) y[i] = (dagger ? std::conj(e[i]) : e[i]) * x[i];
    return y;
}

template<int D>
struct ControlCache {
    int nstep = 0;
    int nb1 = 0;
    std::vector<Mat<D>> r, dx, dy;
    const Mat<D>& R(int g, int n) const { return r[static_cast<std::size_t>(g)*nstep+n]; }
    const Mat<D>& DX(int g, int n) const { return dx[static_cast<std::size_t>(g)*nstep+n]; }
    const Mat<D>& DY(int g, int n) const { return dy[static_cast<std::size_t>(g)*nstep+n]; }
};

template<int D>
ControlCache<D> build_cache(const double* ux, const double* uy, int nstep,
                            const std::vector<double>& b1, double dt) {
    if (nstep < 0 || dt <= 0.0) throw std::invalid_argument("invalid control grid");
    ControlCache<D> out;
    out.nstep = nstep;
    out.nb1 = static_cast<int>(b1.size());
    const std::size_t total = static_cast<std::size_t>(nstep) * b1.size();
    out.r.resize(total); out.dx.resize(total); out.dy.resize(total);

    const auto ix = ix_matrix<D>();
    const auto iy = iy_matrix<D>();
    std::array<double, D*D> ixreal{};
    for (int i = 0; i < D*D; ++i) ixreal[i] = ix[i].real();
    const auto eig = jacobi_real_symmetric<D>(ixreal);
    const auto mz = m_values<D>();

    for (int g = 0; g < out.nb1; ++g) {
        const double bs = b1[g];
        for (int n = 0; n < nstep; ++n) {
            const double x = ux[n], y = uy[n];
            const double amp = std::hypot(x, y);
            const double phi = (amp > 0.0) ? std::atan2(y, x) : 0.0;

            Mat<D> v{}; // eigenvectors of x Ix + y Iy
            for (int i = 0; i < D; ++i) {
                const cd z = std::exp(-Iu * phi * mz[i]);
                for (int k = 0; k < D; ++k) v[mi<D>(i,k)] = z * eig.vec[i*D+k];
            }
            std::array<double,D> lam{};
            std::array<cd,D> ee{};
            for (int k = 0; k < D; ++k) {
                lam[k] = bs * amp * eig.val[k];
                ee[k] = std::exp(-Iu * dt * lam[k]);
            }
            Mat<D> rr{};
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    cd s{0.0,0.0};
                    for (int k = 0; k < D; ++k)
                        s += v[mi<D>(i,k)] * ee[k] * std::conj(v[mi<D>(j,k)]);
                    rr[mi<D>(i,j)] = s;
                }
            }

            auto frechet = [&](const Mat<D>& dh) {
                Mat<D> bmat{};
                for (int k = 0; k < D; ++k) {
                    for (int l = 0; l < D; ++l) {
                        cd s{0.0,0.0};
                        for (int i = 0; i < D; ++i) {
                            for (int j = 0; j < D; ++j)
                                s += std::conj(v[mi<D>(i,k)]) * dh[mi<D>(i,j)] * v[mi<D>(j,l)];
                        }
                        bmat[mi<D>(k,l)] = s;
                    }
                }
                Mat<D> cmat{};
                for (int k = 0; k < D; ++k) {
                    for (int l = 0; l < D; ++l) {
                        cd f;
                        const double dl = lam[k] - lam[l];
                        if (std::abs(dl) < 1e-12 * std::max(1.0, std::abs(lam[k]) + std::abs(lam[l])))
                            f = -Iu * dt * ee[k];
                        else
                            f = (ee[k] - ee[l]) / dl;
                        cmat[mi<D>(k,l)] = f * bmat[mi<D>(k,l)];
                    }
                }
                Mat<D> dr{};
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < D; ++j) {
                        cd s{0.0,0.0};
                        for (int k = 0; k < D; ++k)
                            for (int l = 0; l < D; ++l)
                                s += v[mi<D>(i,k)] * cmat[mi<D>(k,l)] * std::conj(v[mi<D>(j,l)]);
                        dr[mi<D>(i,j)] = s;
                    }
                }
                return dr;
            };

            Mat<D> dhx{}, dhy{};
            for (int i = 0; i < D*D; ++i) { dhx[i] = bs * ix[i]; dhy[i] = bs * iy[i]; }
            const std::size_t idx = static_cast<std::size_t>(g)*nstep+n;
            out.r[idx] = rr;
            out.dx[idx] = frechet(dhx);
            out.dy[idx] = frechet(dhy);
        }
    }
    return out;
}

template<int D>
struct Context {
    int ndrift=0, nphase=0, nens=0, nb1=0, mq_order=3;
    double dt=0.0;
    std::vector<cd> epre, epost; // [drift, phase, D]
    std::vector<double> weight, b1;
    std::vector<int> drift_index, b1_index;
    std::array<double,D> mz = m_values<D>();
    int mq_plus=0, mq_minus=0, ct_plus=0, ct_minus=0;

    const cd* pre(int d, int p) const {
        return epre.data() + (static_cast<std::size_t>(d)*nphase+p)*D;
    }
    const cd* post(int d, int p) const {
        return epost.data() + (static_cast<std::size_t>(d)*nphase+p)*D;
    }
};

template<int D>
inline Vec<D> basis(int i) { Vec<D> v{}; v[i]=1.0; return v; }

template<int D>
inline int phase_at(int offset, int step, int nphase) {
    int p = (offset + step) % nphase;
    if (p < 0) p += nphase;
    return p;
}

template<int D>
inline Vec<D> forward_step(const Context<D>& c, int drift, int p,
                           const Mat<D>& r, const Vec<D>& v) {
    auto a = apply_diag<D>(c.pre(drift,p), v, false);
    a = mat_vec<D>(r, a);
    return apply_diag<D>(c.post(drift,p), a, false);
}

template<int D>
inline Vec<D> backward_step(const Context<D>& c, int drift, int p,
                            const Mat<D>& r, const Vec<D>& v) {
    auto a = apply_diag<D>(c.post(drift,p), v, true);
    a = mat_dag_vec<D>(r, a);
    return apply_diag<D>(c.pre(drift,p), a, true);
}

template<int D>
cd excitation_amplitude(const Context<D>& c, const ControlCache<D>& cc,
                        int s, int offset) {
    Vec<D> x = basis<D>(c.mq_plus);
    Vec<D> y = basis<D>(c.mq_minus);
    const int di=c.drift_index[s], bg=c.b1_index[s];
    for (int n=cc.nstep-1; n>=0; --n) {
        const int p=phase_at<D>(offset,n,c.nphase);
        x=backward_step<D>(c,di,p,cc.R(bg,n),x);
        y=backward_step<D>(c,di,p,cc.R(bg,n),y);
    }
    cd a{0.0,0.0};
    for (int i=0;i<D;++i) a += std::conj(x[i]) * c.mz[i] * y[i];
    return a;
}

template<int D>
cd conversion_amplitude(const Context<D>& c, const ControlCache<D>& cc,
                        int s, int offset) {
    Vec<D> f=basis<D>(c.mq_plus), g=basis<D>(c.mq_minus);
    const int di=c.drift_index[s], bg=c.b1_index[s];
    for (int n=0;n<cc.nstep;++n) {
        const int p=phase_at<D>(offset,n,c.nphase);
        f=forward_step<D>(c,di,p,cc.R(bg,n),f);
        g=forward_step<D>(c,di,p,cc.R(bg,n),g);
    }
    return f[c.ct_plus] * std::conj(g[c.ct_minus]);
}

template<int D>
void excitation_gradient_member(const Context<D>& c, const ControlCache<D>& cc,
                                int s, int offset, cd coeff,
                                std::vector<double>& gx, std::vector<double>& gy) {
    const int N=cc.nstep, di=c.drift_index[s], bg=c.b1_index[s];
    std::vector<Vec<D>> xt(static_cast<std::size_t>(N)+1), yt(static_cast<std::size_t>(N)+1);
    xt[N]=basis<D>(c.mq_plus); yt[N]=basis<D>(c.mq_minus);
    for (int n=N-1;n>=0;--n) {
        const int p=phase_at<D>(offset,n,c.nphase);
        xt[n]=backward_step<D>(c,di,p,cc.R(bg,n),xt[n+1]);
        yt[n]=backward_step<D>(c,di,p,cc.R(bg,n),yt[n+1]);
    }
    Vec<D> pv{}, qv{};
    for (int i=0;i<D;++i) { pv[i]=c.mz[i]*yt[0][i]; qv[i]=c.mz[i]*xt[0][i]; }
    for (int n=0;n<N;++n) {
        const int ph=phase_at<D>(offset,n,c.nphase);
        const cd* ep=c.pre(di,ph); const cd* eo=c.post(di,ph);
        const auto ap=apply_diag<D>(ep,pv,false);
        const auto aq=apply_diag<D>(ep,qv,false);
        const auto bx=apply_diag<D>(eo,xt[n+1],true);
        const auto by=apply_diag<D>(eo,yt[n+1],true);
        auto contract=[&](const Mat<D>& dr){
            const cd t1=inner<D>(bx,mat_vec<D>(dr,ap));
            const cd t2=inner<D>(by,mat_vec<D>(dr,aq));
            return t1 + std::conj(t2);
        };
        gx[n] += 2.0 * std::real(coeff * contract(cc.DX(bg,n)));
        gy[n] += 2.0 * std::real(coeff * contract(cc.DY(bg,n)));
        pv=forward_step<D>(c,di,ph,cc.R(bg,n),pv);
        qv=forward_step<D>(c,di,ph,cc.R(bg,n),qv);
    }
}

template<int D>
void conversion_gradient_member(const Context<D>& c, const ControlCache<D>& cc,
                                int s, int offset, cd coeff,
                                std::vector<double>& gx, std::vector<double>& gy) {
    const int N=cc.nstep, di=c.drift_index[s], bg=c.b1_index[s];
    std::vector<Vec<D>> ft(static_cast<std::size_t>(N)+1), gt(static_cast<std::size_t>(N)+1);
    std::vector<Vec<D>> xt(static_cast<std::size_t>(N)+1), yt(static_cast<std::size_t>(N)+1);
    ft[0]=basis<D>(c.mq_plus); gt[0]=basis<D>(c.mq_minus);
    for (int n=0;n<N;++n) {
        const int p=phase_at<D>(offset,n,c.nphase);
        ft[n+1]=forward_step<D>(c,di,p,cc.R(bg,n),ft[n]);
        gt[n+1]=forward_step<D>(c,di,p,cc.R(bg,n),gt[n]);
    }
    xt[N]=basis<D>(c.ct_plus); yt[N]=basis<D>(c.ct_minus);
    for (int n=N-1;n>=0;--n) {
        const int p=phase_at<D>(offset,n,c.nphase);
        xt[n]=backward_step<D>(c,di,p,cc.R(bg,n),xt[n+1]);
        yt[n]=backward_step<D>(c,di,p,cc.R(bg,n),yt[n+1]);
    }
    const cd alpha=ft[N][c.ct_plus];
    const cd beta=gt[N][c.ct_minus];
    for (int n=0;n<N;++n) {
        const int ph=phase_at<D>(offset,n,c.nphase);
        const cd* ep=c.pre(di,ph); const cd* eo=c.post(di,ph);
        const auto af=apply_diag<D>(ep,ft[n],false);
        const auto ag=apply_diag<D>(ep,gt[n],false);
        const auto bx=apply_diag<D>(eo,xt[n+1],true);
        const auto by=apply_diag<D>(eo,yt[n+1],true);
        auto contract=[&](const Mat<D>& dr){
            const cd da=inner<D>(bx,mat_vec<D>(dr,af));
            const cd db=inner<D>(by,mat_vec<D>(dr,ag));
            return std::conj(beta)*da + alpha*std::conj(db);
        };
        gx[n] += 2.0 * std::real(coeff * contract(cc.DX(bg,n)));
        gy[n] += 2.0 * std::real(coeff * contract(cc.DY(bg,n)));
    }
}

template<int D>
int evaluate_pair(Context<D>& c,
                  const double* ex, const double* ey, int ne, int offe,
                  const double* cx, const double* cy, int nc, int offc,
                  double* metrics,
                  double* gex, double* gey, double* gcx, double* gcy) {
    try {
        auto ce=build_cache<D>(ex,ey,ne,c.b1,c.dt);
        auto cc=build_cache<D>(cx,cy,nc,c.b1,c.dt);
        std::vector<cd> ae(c.nens), ac(c.nens);
#pragma omp parallel for schedule(static)
        for (int s=0;s<c.nens;++s) {
            ae[s]=excitation_amplitude<D>(c,ce,s,offe);
            ac[s]=conversion_amplitude<D>(c,cc,s,offc);
        }
        cd C{0.0,0.0}, mean_e{0.0,0.0}, mean_c{0.0,0.0};
        double inc=0.0, abse=0.0, absc=0.0;
        for (int s=0;s<c.nens;++s) {
            const double w=c.weight[s];
            C += w*ae[s]*ac[s]; mean_e += w*ae[s]; mean_c += w*ac[s];
            inc += w*std::abs(ae[s]*ac[s]); abse += w*std::abs(ae[s]); absc += w*std::abs(ac[s]);
        }
        const double spin=0.5*static_cast<double>(D-1);
        const double norm2=spin*spin;
        const double J=std::norm(C)/norm2;
        if (metrics) {
            metrics[0]=J;
            metrics[1]=C.real()/spin; metrics[2]=C.imag()/spin;
            metrics[3]=std::abs(C)/spin;
            metrics[4]=inc/spin;
            metrics[5]=(inc>0.0?std::abs(C)/inc:0.0);
            metrics[6]=std::abs(mean_e)/spin;
            metrics[7]=std::abs(mean_c);
            metrics[8]=abse/spin;
            metrics[9]=absc;
        }
        if (!gex || !gey || !gcx || !gcy) return 0;
        std::fill(gex,gex+ne,0.0); std::fill(gey,gey+ne,0.0);
        std::fill(gcx,gcx+nc,0.0); std::fill(gcy,gcy+nc,0.0);
#pragma omp parallel
        {
            std::vector<double> lex(ne,0.0), ley(ne,0.0), lcx(nc,0.0), lcy(nc,0.0);
#pragma omp for schedule(static)
            for (int s=0;s<c.nens;++s) {
                const cd coeffe=std::conj(C)*c.weight[s]*ac[s]/norm2;
                const cd coeffc=std::conj(C)*c.weight[s]*ae[s]/norm2;
                excitation_gradient_member<D>(c,ce,s,offe,coeffe,lex,ley);
                conversion_gradient_member<D>(c,cc,s,offc,coeffc,lcx,lcy);
            }
#pragma omp critical
            {
                for(int n=0;n<ne;++n){gex[n]+=lex[n];gey[n]+=ley[n];}
                for(int n=0;n<nc;++n){gcx[n]+=lcx[n];gcy[n]+=lcy[n];}
            }
        }
        return 0;
    } catch (...) { return -1; }
}

template<int D>
Context<D>* create_context(int ndrift,int nphase,int nens,int nb1,double dt,int mq_order,
                           const double* epre_ri,const double* epost_ri,
                           const double* weight,const int* didx,const int* bidx,
                           const double* b1) {
    if(ndrift<=0||nphase<=0||nens<=0||nb1<=0||dt<=0.0) throw std::invalid_argument("invalid context size");
    auto c=std::make_unique<Context<D>>();
    c->ndrift=ndrift;c->nphase=nphase;c->nens=nens;c->nb1=nb1;c->dt=dt;c->mq_order=mq_order;
    const std::size_t nz=static_cast<std::size_t>(ndrift)*nphase*D;
    c->epre.resize(nz);c->epost.resize(nz);
    for(std::size_t i=0;i<nz;++i){c->epre[i]=cd(epre_ri[2*i],epre_ri[2*i+1]);c->epost[i]=cd(epost_ri[2*i],epost_ri[2*i+1]);}
    c->weight.assign(weight,weight+nens);c->drift_index.assign(didx,didx+nens);c->b1_index.assign(bidx,bidx+nens);c->b1.assign(b1,b1+nb1);
    double ws=std::accumulate(c->weight.begin(),c->weight.end(),0.0); if(ws<=0.0) throw std::invalid_argument("weights must sum positive");
    for(double& w:c->weight)w/=ws;
    const double spin=0.5*(D-1);
    const double mp=0.5*mq_order, mm=-mp;
    auto find_idx=[&](double target){for(int i=0;i<D;++i)if(std::abs(c->mz[i]-target)<1e-9)return i;return -1;};
    c->mq_plus=find_idx(mp);c->mq_minus=find_idx(mm);c->ct_plus=find_idx(0.5);c->ct_minus=find_idx(-0.5);
    if(mq_order<1||mq_order%2==0||mq_order>2*spin||c->mq_plus<0||c->mq_minus<0)throw std::invalid_argument("invalid odd MQ order");
    for(int s=0;s<nens;++s){if(c->drift_index[s]<0||c->drift_index[s]>=ndrift||c->b1_index[s]<0||c->b1_index[s]>=nb1)throw std::invalid_argument("mapping index out of range");}
    return c.release();
}

} // namespace mqmas

extern "C" {

void* mqmas_create_d4(int ndrift,int nphase,int nens,int nb1,double dt,int mq_order,
                      const double* epre,const double* epost,const double* w,
                      const int* didx,const int* bidx,const double* b1) {
    try{return mqmas::create_context<4>(ndrift,nphase,nens,nb1,dt,mq_order,epre,epost,w,didx,bidx,b1);}catch(...){return nullptr;}
}
void* mqmas_create_d6(int ndrift,int nphase,int nens,int nb1,double dt,int mq_order,
                      const double* epre,const double* epost,const double* w,
                      const int* didx,const int* bidx,const double* b1) {
    try{return mqmas::create_context<6>(ndrift,nphase,nens,nb1,dt,mq_order,epre,epost,w,didx,bidx,b1);}catch(...){return nullptr;}
}
void mqmas_destroy_d4(void* p){delete static_cast<mqmas::Context<4>*>(p);} 
void mqmas_destroy_d6(void* p){delete static_cast<mqmas::Context<6>*>(p);} 

int mqmas_eval_pair_d4(void* p,const double* ex,const double* ey,int ne,int oe,
                       const double* cx,const double* cy,int nc,int oc,double* metrics,
                       double* gex,double* gey,double* gcx,double* gcy){
    if(!p)return -2;return mqmas::evaluate_pair<4>(*static_cast<mqmas::Context<4>*>(p),ex,ey,ne,oe,cx,cy,nc,oc,metrics,gex,gey,gcx,gcy);
}
int mqmas_eval_pair_d6(void* p,const double* ex,const double* ey,int ne,int oe,
                       const double* cx,const double* cy,int nc,int oc,double* metrics,
                       double* gex,double* gey,double* gcx,double* gcy){
    if(!p)return -2;return mqmas::evaluate_pair<6>(*static_cast<mqmas::Context<6>*>(p),ex,ey,ne,oe,cx,cy,nc,oc,metrics,gex,gey,gcx,gcy);
}

}
