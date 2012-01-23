/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef LL_H
#define LL_H

struct llitem {
	void *data;
	llitem *next;
};

struct delptr {
	void *ptr;
};

class ll {
	public:
		ll() {
			head = NULL;
			iter = NULL;
		}

		virtual ~ll() {
		}

		void ins(void *data) {
			llitem *newitem = (llitem *)malloc(sizeof(llitem));
			newitem->data = data;
			newitem->next = head;
			head = newitem;
		}

		void del(void *data) {
			llitem *cur = head;
			llitem *prev = NULL;
			while (cur && cur->data != data) {
				prev = cur;
				cur = cur->next;
			}
			if (cur) { // cur is our item to remove, its data ptr matches the input
				if (prev)
					prev->next = cur->next;
				if (head == cur)
					head = cur->next;
				free(cur);
			}
		}

		void *del2(void *ptr) {
			llitem *cur = head;
			llitem *prev = NULL;
			while (cur && ((delptr *)(cur->data))->ptr != ptr) {
				prev = cur;
				cur = cur->next;
			}
			if (cur) { // cur is our item to remove, its data ptr matches the input
				if (prev)
					prev->next = cur->next;
				if (head == cur)
					head = cur->next;
				void *data = cur->data;
				free(cur);
				return data;
			}
			return NULL;
		}

		void toHead() {
			iter = head;
		}
		void *getNext() {
			if (iter) {
				void *data = iter->data;
				iter = iter->next;
				return data;
			} else {
				return NULL;
			}
		}

	protected:
		llitem *head;
		llitem *iter;
};

#endif // LL_H

