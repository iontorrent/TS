/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 *  CircularQueue.h
 *  SamUtils_pileup
 *
 *  Created by Michael Lyons on 5/4/11.
 *
 */


#ifndef CIRCULARQUEUE_H
#define CIRCULARQUEUE_H



template<typename element, unsigned int size>
/** 
 A Circular Queue  
 Thread safe for one reader, and one writer 
 Queue isn't dynamic.  It recieves a static initialization size and uses that to 
 ensure thread safety.  
 
 Initialized with a typename and size
 */
class CircularQueue {
public:
	enum {Capacity = size+1};
	/**
	 Constructor which initializes the queue.  
	 
	 */
	CircularQueue() : tail(0), head(0){}
	virtual ~CircularQueue() {}
	
	/**
	 Adds an element to the queue.  Returns false if the queue is full
	 
	 @param	element& _item		an element to be added to the queue
	 @return	bool			returns true if element successfully added to queue
	 */
	bool put(element& _item){ 
		unsigned int next_tail = increment(tail);
		if(next_tail != head)
		{
			array[tail] = _item;
			tail = next_tail;
			return true;
		}
		
		// queue was full
		return false;
	}
	/**
	 tries to pop an element from the queue.  returns false if the queue is empty.
	 Overwrites the contents of element& _item, and that is the return'ed element
	 from the queue.
	 
	 @param	element& _item		this is the variable that contains the returned item from the queue
	 @return	bool			returns true if pop was successfull, false otherwise
	 */
	bool pop(element& _item){ 
		
		if(head == tail)
			return false;  // empty queue
		
		_item = array[head];
		head = increment(head);
		return true;
	}
	
	/**
	 Convenience function to check if the queue is empty
	 
	 @return	bool		true if the queue is empty
	 */
	bool is_empty() const {
		return (head == tail);

	}
	/**
	 Convenience function to check if the queue is full
	 
	 @return	bool		true if the queue is full
	 */
	bool is_full() const{ 
		unsigned int tail_check = (tail+1) % Capacity;
		return (tail_check == head);

	}
	
	/**
	 * non-thread safe function, but will help in load balancing.  It's more of an "estimate"
	 * than anything :)
	 *
	 *@return total capacity
	 */
	unsigned int get_size() const {
		unsigned int size_return = tail - head;

		return size_return;

	}

private:
	volatile unsigned int tail; // input index
	element array[Capacity];
	volatile unsigned int head; // output index
	
	unsigned int increment(unsigned int _idx) const {
		
		_idx = (_idx+1) % Capacity;
		return _idx;

		
	}
};






#endif // CIRCULARQUEUE_H
