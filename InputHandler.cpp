#include "InputManager.hpp"
#include "InputHandler.hpp"

using namespace clouds;

template<typename T>
InputHandler<T>::InputHandler(Handler h) :
    handler(h),
    name("<anonymous>"),
    parent(nullptr),
    handler_id(0)
    {}

template<typename T>
InputHandler<T>::InputHandler(const std::string& n, Handler h) :
    handler(h),
    name(n.empty() ? std::string("<anonymous>") : n),
    parent(nullptr),
    handler_id(0)
    {}

template<typename T>
InputHandler<T>::InputHandler(InputHandler&& other) :
    handler(std::move(other.handler)),
    name(std::move(other.name)),
    parent(other.parent),
    handler_id(other.handler_id)
{
    other.parent = nullptr;
}

template<typename T>
void InputHandler<T>::registerSelf(InputManager* mgr, int id)
{
    parent = mgr;
}

template<typename T>
InputHandler<T>::~InputHandler()
{
    if(parent && handler_id)
    {
        parent->removeHandler(handler_id);
    }
}
