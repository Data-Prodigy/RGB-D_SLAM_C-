#pragma once

#include "../src/pch.h"

namespace sfs {

    template<typename T>
    class ThreadSafeQueue {
    public:
        void push(const T& item) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(item);
            cond_.notify_one();
        }

        void push(T&& item) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
            cond_.notify_one();
        }

        bool pop(T& item) {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this] { return !queue_.empty() || finished_; });
            if (queue_.empty()) return false;
            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }

        void setFinished() {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
            cond_.notify_all();
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

    private:
        mutable std::mutex mutex_;
        std::queue<T> queue_;
        std::condition_variable cond_;
        bool finished_ = false;
    };

} // namespace sfs