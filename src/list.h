#ifndef LIST_H
#define LIST_H

/*
A typedef, in spite of the name, 
does not define a new type; it merely creates a new name for an existing type
*/

typedef struct node{
    /*

    void pointer or a generic pointer is a special type of pointer that
    can be pointed at objects of any data type

    */
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);

void free_list_contents(list *l);
void **list_to_array(list *l);
void free_list(list *l);

#endif
