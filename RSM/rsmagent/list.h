#ifndef LIST_H
#define LIST_H

typedef struct singlyLinkedList_s
{
	struct singlyLinkedList_s *next;
	void *item;
} list_t;

list_t *list_create();
int     list_end(list_t *listHead, list_t **endPtr);
list_t *list_nextItem(list_t const * const lptr);
void    list_insert_after(list_t *lptr, void *newItem);
void    list_append(list_t *listHead, void *newItem);
list_t *list_prepend(list_t *listHead, void *newItem);
void    list_delete_after(list_t *item);
list_t *list_delete_head(list_t *listHead);
void    list_free(list_t *listHead);

#endif	// LIST_H
