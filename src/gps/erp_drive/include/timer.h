#ifndef TIMEMR_H
#define TIMER_H

#include "ros/ros.h"

enum TimerState {
    INIT,
    STARTED,
    DONE
};

class Timer {
public:
    Timer(const double& _target_time){
        this->target_time = _target_time;
        this->state = TimerState::INIT;
    }

    void resetTimer(const double& _target_time){
        this->target_time = _target_time;
        this->state = TimerState::INIT;
    }

    void start(){
        if(this->state != TimerState::INIT) return;

        this->state = TimerState::STARTED;
        this->basis_time = ros::Time::now();
    }

    bool isTimeout(){
        if(this->state != TimerState::STARTED)
            return true;

        if(this->getTimeSpan() > this->target_time){
            this->state = TimerState::DONE;
            return true;
        }

        return false;
    }

    bool isDone(){
        return this->state == TimerState::DONE;
    }

    double getTimeSpan() const {
        return (ros::Time::now() - this->basis_time).toNSec() / 1e9;
    }

    double getTargetTime() const {
        return target_time;
    }

private:
    TimerState state; // 0: init, 1: start, 2: end
    double target_time;
    ros::Time basis_time;
};

#endif