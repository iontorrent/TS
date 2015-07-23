#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct singlyLinkedList_s
{
	struct singlyLinkedList_s *next;
	void *item;
} list_t;

list_t *list_create()
{
	list_t *listHead = malloc(sizeof(list_t));
	if (listHead) {
		listHead->item = NULL;
		listHead->next = NULL;
	}
	return listHead;
}

int list_end(list_t *listHead, list_t **endPtr)
{
	if (!listHead) {
		*endPtr = NULL;
		return 0;
	}

	int count = 0;
	list_t *lptr = listHead;

	while (lptr->next) {
		lptr = lptr->next;
		++count;
	}

	if (endPtr)
		*endPtr = lptr;
	return count;
}

list_t *list_nextItem(list_t const * const lptr)
{
	return lptr->next;
}

void list_insert_after(list_t *lptr, void *newItem)
{
	if (!lptr || !newItem)
		return;

	list_t *newPtr = malloc(sizeof(list_t));
	if (newPtr) {
		list_t *next = lptr->next;
		lptr->next = newPtr;
		newPtr->item = newItem; // save pointer to existing item; we don't know how big it is nor how to copy it correctly
		newPtr->next = next;
	}
}

void list_append(list_t *listHead, void *newItem)
{
	if (!listHead || !newItem)
		return;
	list_t *lptr;
	list_end(listHead, &lptr);
	list_insert_after(lptr, newItem);
}

list_t *list_prepend(list_t *listHead, void *newItem)
{
	if (!listHead || !newItem)
		return listHead;

	list_t *newPtr = malloc(sizeof(list_t));
	if (newPtr) {
		newPtr->next = listHead;
		newPtr->item = newItem; // save pointer to existing item; we don't know how big it is nor how to copy it correctly
		listHead = newPtr;
	}

	return listHead;
}

void list_delete_after(list_t *item)
{
	if (!item)
		return;

	list_t *lose = item->next;
	if (lose) {
		item->next = lose->next;
		memset(lose, 0, sizeof *lose);
		free(lose);
	}
}

list_t *list_delete_head(list_t *listHead)
{
	if (!listHead)
		return NULL;

	list_t *newHead = listHead->next;
	memset(listHead, 0, sizeof *listHead);
	free(listHead);
	return newHead;
}

void list_free(list_t *listHead)
{
	if (!listHead)
		return;

	list_t *lptr = listHead;
	while (lptr) {
		list_t *next = lptr->next;
		memset(lptr, 0, sizeof *lptr);
		free(lptr);
		lptr = next;
	}
}

typedef struct sample_s
{
	char *str;
} sample_t;

//#define LIST_UNIT_TEST
#ifdef LIST_UNIT_TEST

void print_list(list_t const * const listIn)
{
	list_t *lptr = (list_t *)listIn;
	while (lptr) {
		sample_t *s = (sample_t *)(lptr->item);
		if (s) {
			char *str = s->str;
			printf("%s\n", str ? str : "NULL");
		}
		else {
			printf("NULL item\n");
		}
		lptr = list_nextItem(lptr);
	}
	printf("\n");
}

int main(void)
{
	sample_t s1 = {"s1"};
	sample_t s2 = {"s2"};
	sample_t s3 = {"s3"};
	sample_t s4 = {"s4"};

	list_t *hosts = list_create();
	hosts->item = &s1;
	list_append(hosts, &s2);
	list_append(hosts, &s3);
	list_append(hosts, &s4);
	print_list(hosts);

	sample_t s0 = {"s0"};
	hosts = list_prepend(hosts, &s0);
	print_list(hosts);

	sample_t s2a = {"s2a"};
	list_t *lptr = hosts->next; // 1
	lptr = lptr->next;			// 2
	list_insert_after(lptr, &s2a);
	print_list(hosts);

	hosts = list_delete_head(hosts);
	print_list(hosts);

	lptr = hosts->next; // 2
	list_delete_after(lptr);
	print_list(hosts);

	list_free(hosts);
	hosts = NULL;

	return 0;
}
#endif
