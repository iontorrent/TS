/* Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved */
#ifndef RINGBUFFER_H
#define RINGBUFFER_H

template <class RingTemplate>
class RingBuffer {

public:
	const static int ARRAY_SIZE = 32;

	RingBuffer() {
		head = 0;
		tail = 0;
	}
	bool insert(RingTemplate n) {
		if (!isFull()) {
			myArray[head] = n;
			head = (head + 1) % ARRAY_SIZE;
			return true;
		}
		return false;
	}
	RingTemplate remove() {
		int oldTail = tail;
		if (!isEmpty()) {
			tail = (tail + 1) % ARRAY_SIZE;
			return myArray[oldTail];
		}
		return NULL;
	}

	bool isFull() {
		return ((head + 1) % ARRAY_SIZE == tail);
	}
	bool isEmpty() {
		return (tail == head);
	}
	int remaining() {
		if (head >= tail) {
			return ARRAY_SIZE - head + tail;
		}
		return tail - head - 1;
	}

private:
	RingTemplate myArray[ARRAY_SIZE];

	//insert at head...
	int head;
	//...remove at tail
	int tail;
};

#endif // RINGBUFFER_H
