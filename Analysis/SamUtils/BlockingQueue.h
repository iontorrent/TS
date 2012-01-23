/* Copyright (C) 2011 Ion Torrent Systems, Inc. All Rights Reserved */
/*
 *  BlockingQueue.h
 *  SamUtils_mt_branch
 *
 *  Created by Michael Lyons on 2/22/11.
 *  Copyright 2011 Life Technologies. All rights reserved.
 *
 */

#ifndef BLOCKINGQUEUE_H
#define BLOCKINGQUEUE_H

#include "Lock.h"
#include <queue>

template<typename T> class BlockingQueue {
public:
    
    virtual void put(T o) = 0;
    virtual int remaining_capacity() = 0;
    virtual T pop() = 0;
    virtual bool is_empty() = 0;
    virtual void set_done(bool done) = 0;
	virtual bool am_done() = 0;
    virtual ~BlockingQueue() { };
};

template<typename T> class SimpleBlockingQueue : public BlockingQueue<T> {
    unsigned int capacity;
    std::queue<T> queue_;
    NativeMutex queue_mutex;
    NativeCond cond_full;
    NativeCond cond_empty;
	bool	   im_done;
	
public:
    
	SimpleBlockingQueue() {
		capacity = 0;
	}
	
	
    SimpleBlockingQueue(int Capacity) : capacity(Capacity) {
		im_done = false;
    }
	
	void set_capacity(int Capacity) {
		capacity = Capacity;
	}
	
	virtual void put(T o) {
        {
			//std::cerr<< "[BlockingQueue - void put(T o)] queue_.size(): " << queue_.size() << " capacity: " << capacity<<std::endl;
			//std::cerr << "[BlockingQueue - void put(T o)] acquiring lock"<<std::endl;

            ScopedLock lck(&queue_mutex);
			//std::cerr << "[BlockingQueue - void put(T o)] acquired lock"<<std::endl;
            while(queue_.size() > capacity)
                cond_full.wait(&queue_mutex);
            queue_.push(o);
        }
        cond_empty.broadcast();
	}
	
    virtual T pop() {

			T value;
			{
				//std::cerr << "[BlockingQueue - T pop()] acquiring lock"<<std::endl;
				
				ScopedLock lck(&queue_mutex);
				//std::cerr << "[BlockingQueue - T pop()] acquired lock"<<std::endl;

				while(queue_.size() == 0) {
					cond_empty.wait(&queue_mutex);
				}
				value = queue_.front();
				queue_.pop();
				
			}
			cond_full.broadcast();
			return value;
		
		
    }
    
    virtual int remaining_capacity() {
		//std::cerr << "[BlockingQueue - int remaining_capacity()] acquiring lock"<<std::endl;

        ScopedLock lck(&queue_mutex);
		//std::cerr << "[BlockingQueue - int remaining_capacity()] acquired lock"<<std::endl;
		//std::cerr << "[BlockingQueue - int remaining_capacity()] capacity: "<< capacity << " queue_.size(): " << queue_.size()<< std::endl;

        int remaining = capacity - queue_.size();
        return remaining;
    }
	
    virtual bool is_empty() {

        bool rv;
		{   //forcing some scope on this      
			//std::cerr << "[BlockingQueue - bool is_empty()] acquiring lock"<<std::endl;

			ScopedLock lck(&queue_mutex);
			//std::cerr << "[BlockingQueue - bool is_empty()] acquired lock"<<std::endl;

			rv= queue_.empty();
		}
		//std::cerr << "[BlockingQueue - bool is_empty()] lock destroyed"<<std::endl;

        return rv;
    }
	
	virtual void set_done(bool done) {
		ScopedLock lck(&queue_mutex);
		im_done = done;
	}
	
	virtual bool am_done() {
		ScopedLock lck(&queue_mutex);
		return im_done;
	}
    
    virtual ~SimpleBlockingQueue() { }
};

#endif // BLOCKINGQUEUE_H

