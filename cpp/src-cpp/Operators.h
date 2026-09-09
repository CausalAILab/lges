//
// Created by Achille Nazaret on 11/4/23.
//

#pragma once


#include "set_ops.h"
#include <iostream>

#include "spdlog/fmt/ostr.h"

class Reverse;


class Insert {
public:
    Insert(int x, int y, FlatSet T, double score, FlatSet effective_parents);

    friend std::ostream &operator<<(std::ostream &os, const Insert &obj);
    friend std::ostream &operator<<(std::ostream &os, const Reverse &obj);

    friend class PDAG;

    friend class XGES;

    bool operator<(const Insert &rhs) const { return score < rhs.score; }

    int get_x() const { return x; }
    int get_y() const { return y; }
    const FlatSet &get_T() const { return T; }
    double get_score() const { return score; }
    const FlatSet &get_effective_parents() const { return effective_parents; }

private:
    int x, y;
    FlatSet T;
    double score = 0;
    FlatSet effective_parents;// = [Ne(y) ∩ Ad(x)] ∪ T ∪ Pa(y)
};

class Delete {
public:
    Delete(int x, int y, FlatSet C, double score, FlatSet effective_parents,
           bool directed);

    friend std::ostream &operator<<(std::ostream &os, const Delete &obj);

    friend class PDAG;

    friend class XGES;

    bool operator<(const Delete &rhs) const { return score < rhs.score; }


private:
    int x, y;
    FlatSet C;
    double score = 0;
    FlatSet effective_parents;// C ∪ Pa(y) ∪ {x}
    bool directed;
};


class Reverse {
public:
    Reverse(Insert insert, double score, FlatSet parents_x);

    friend std::ostream &operator<<(std::ostream &os, const Reverse &obj);

    friend class PDAG;

    friend class XGES;

    bool operator<(const Reverse &rhs) const { return score < rhs.score; }


private:
    Insert insert;
    double score = 0;
    FlatSet parents_x;
};

// fmt >= 9 (bundled in spdlog) requires opting in to ostream-based formatting.
template<>
struct fmt::formatter<Insert> : fmt::ostream_formatter {};
template<>
struct fmt::formatter<Delete> : fmt::ostream_formatter {};
template<>
struct fmt::formatter<Reverse> : fmt::ostream_formatter {};
