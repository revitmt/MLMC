/*
 * my_timer.h
 *
 *  Created on: Feb 15, 2015
 *      Author: viktor
 */

#ifndef MY_TIMER_H_
#define MY_TIMER_H_


#include <ctime>
#include <ratio>
#include <chrono>

using namespace std::chrono;


class my_timer
{
	public:
		my_timer(){};
		virtual ~my_timer(){};
		void tic()
		{
			t1 = high_resolution_clock::now();
		};
		double toc()
		{
			t2 = high_resolution_clock::now();
			time_span = duration_cast<duration<double>>(t2 - t1);

			return time_span.count();
		};

	private:
		duration<double> time_span;
		high_resolution_clock::time_point t1, t2;
};



#endif /* MY_TIMER_H_ */
