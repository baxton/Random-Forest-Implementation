

#if !defined HEAP_DOT_H
#define HEAP_DOT_H

#include <stdlib.h>

struct heap_node {
    double key;
    void* data;
};


struct heap {
    struct heap_node* array;
    int size;
    int num_items;
};


struct heap* allocate_heap(int size);
void free_heap(struct heap* h);

int heap_size(struct heap* h);

struct heap_node* heap_top(struct heap* h);
void heap_add(struct heap* h, double key, void* data);
void heap_pop(struct heap* h);



#endif  // HEAP_DOT_H
