
#include <stdio.h>
#include <string.h>

#include <heap.h>

//
// gcc -g -I. -DAS_LIB -c heap.c -o heap.o
//

struct heap* allocate_heap(int size) {
    struct heap* tmp = (struct heap*)malloc(sizeof(struct heap));
    tmp->array = (struct heap_node*)malloc(size * sizeof(struct heap_node));
    memset(tmp->array, 0, size * sizeof(struct heap_node));
    tmp->size = size;
    tmp->num_items = 0;

    return tmp;
}

void free_heap(struct heap* p) {
    free(p->array);
    free(p);
}


int get_parent(int idx) {
    if (0 >= idx)
        return -1;
    else if (1 % idx)
        return (idx - 1) / 2;
    else
        return (idx - 2) / 2;
}

int get_left(int idx) {
    return idx * 2 + 1;
}

int get_right(int idx) {
    return idx * 2 + 2;
}


struct heap_node* heap_top(struct heap* h) {
    if (h->num_items)
        return &h->array[0];
    return NULL;
}

int heap_size(struct heap* h) {
    return h->num_items;
}

int heap_capacity(struct heap* h) {
    return h->size;
}

int heap_get_node_by_value(struct heap* h, double key, void* data) {
    long len = heap_size(h);
    for (int i = 0; i < len; ++i) {
        struct heap_node* n = &h->array[i];
        if (n->data == data) {
            return i;
        }
    }

    return -1;
}

void heap_add(struct heap* h, double key, void* data) {
    struct heap_node* n;
    long idx = heap_get_node_by_value(h, key, data);
    if (idx < 0) {
        if (h->num_items == h->size) {
            if (key > heap_top(h)->key)
                heap_pop(h);
            else
                return;
        }

        idx = h->num_items;
        ++h->num_items;
        n = &h->array[idx];
    }else{
        n = &h->array[idx];
        if (key <= n->key)
            return;
    }

    n->key = key;
    n->data = data;

    int parent = get_parent(idx);

    while (0 <= parent && h->array[parent].key > n->key) {
        struct heap_node tmp;
        memcpy(&tmp, &h->array[parent], sizeof(struct heap_node));
        memcpy(&h->array[parent], n, sizeof(struct heap_node));
        memcpy(n, &tmp, sizeof(struct heap_node));

        n = &h->array[parent];
        parent = get_parent(parent);
    }
}

void heap_pop(struct heap* h) {
    if (!h->num_items)
        return;

    if (1 == h->num_items) {
        --h->num_items;
        return;
    }

    int last_idx = h->num_items - 1;
    memcpy(&h->array[0], &h->array[last_idx], sizeof(struct heap_node));
    --h->num_items;

    int cur_idx = 0;
    int left = get_left(cur_idx);
    int right = get_right(cur_idx);

    if (left >= last_idx && right >= last_idx)
        return;

    int smallest = (left < last_idx && right < last_idx) ?
                        (h->array[left].key < h->array[right].key ? left : right) :
                        (left < last_idx ? left : right);

    while (h->array[cur_idx].key > h->array[smallest].key) {
        struct heap_node tmp;
        memcpy(&tmp, &h->array[cur_idx], sizeof(struct heap_node));
        memcpy(&h->array[cur_idx], &h->array[smallest], sizeof(struct heap_node));
        memcpy(&h->array[smallest], &tmp, sizeof(struct heap_node));

        cur_idx = smallest;
        left = get_left(cur_idx);
        right = get_right(cur_idx);

        if (left >= last_idx && right >= last_idx)
            break;

        smallest = (left < last_idx && right < last_idx) ?
                        (h->array[left].key < h->array[right].key ? left : right) :
                        (left < last_idx ? left : right);
    }
}








#if !defined AS_LIB

int main() {

    printf("Constructing heap\n");
    struct heap* h = allocate_heap(5);
    heap_add(h, 5, "maxim");
    heap_add(h, 4, "elya");
    heap_add(h, 1, "vika");
    heap_add(h, 3, "lera");
    heap_add(h, 1, "misha");

    heap_add(h, 2, "hammy");

    printf("Printing heap: %d items\n", heap_size(h));
    while (heap_size(h)) {
        struct heap_node* n = heap_top(h);
        printf("%f %s\n", n->key, (const char*)n->data);
        heap_pop(h);
    }

    free_heap(h);

    //
    h = allocate_heap(5);
    heap_add(h, 0.45, (void*)25);
    heap_add(h, 0.41, (void*)22);
    heap_add(h, 0.42, (void*)40033);
    heap_add(h, 0.47, (void*)38919);
    heap_add(h, 0.47, (void*)2215);

    printf("Printing heap: %d items\n", heap_size(h));
    while (heap_size(h)) {
        struct heap_node* n = heap_top(h);
        printf("%f %d\n", n->key, (int)n->data);
        heap_pop(h);
    }

    free_heap(h);

    return 0;
}

#endif




