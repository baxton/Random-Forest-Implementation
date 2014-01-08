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

void heap_add(struct heap* h, double key, void* data) {
    if (h->num_items == h->size)
        return;

    struct heap_node* n = &h->array[h->num_items];
    n->key = key;
    n->data = data;
    ++h->num_items;

    int idx = h->num_items - 1;
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

    printf("Constructin heap\n");
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
        printf("%5.2f %s\n", n->key, (const char*)n->data);
        heap_pop(h);
    }

    free_heap(h);

    return 0;
}

#endif


