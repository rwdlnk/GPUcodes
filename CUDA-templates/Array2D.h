//Array2D.h

#ifndef ARRAY2D_H
#define ARRAY2D_H

template <class T>
class Array2D {
public:
    Array2D(T* _data,
            const size_t& _nrows,
            const size_t& _ncols);
    Array2D(const Array2D<T>& other);
    Array2D<T>& operator=(const Array2D<T>& other);
    ~Array2D(){delete[] this->data;}
    size_t get_nrows() const {return this->nrows;}
    size_t get_ncols() const {return this->ncols;}
    size_t size()      const {return this->N;}
    T* begin(){return data;}
    T* begin()const{return data;}
    T* end(){return data + this->size();}
    T* end()const{return data + this->size();}
private:
    T* data;
    size_t nrows;
    size_t ncols;
    size_t N;

};

template <class T>
Array2D<T>::Array2D(T* _data,
                    const size_t& _nrows,
                    const size_t& _ncols):data(_data), nrows(_nrows), ncols(_ncols){
    this->N = _nrows * _ncols;
};


template <class T>
Array2D<T>::Array2D(const Array2D<T>& other):nrows(other.nrows), ncols(other.ncols), N(other.N){
    data = new T[N];
    auto i = this->begin();
    for (auto& o:other)*i++=o;
};


template <class T>
Array2D<T>& Array2D<T>::operator=(const Array2D<T>& other){
    this->ncols = other.ncols;
    this->ncols = other.nrows;
    this->N     = other.N;

    // here should compare the sizes of the arrays and reallocate if necessary
    delete[] data;
    data = new T[N];
    auto i = this->begin();
    for (auto& o:other)*i++=o;
    return *this;
};

#endif //ARRAY2D_H
