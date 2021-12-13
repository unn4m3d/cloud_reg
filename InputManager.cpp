#include "InputManager.hpp"
#include "InputHandler.hpp"

using namespace clouds;

InputManager::InputManager() :
    current_handler_id(1)
    {}

InputManager::HandlerID InputManager::addHandler(InputHandlerBase&& handle)
{
    handle.registerSelf(this, current_handler_id);
    handlers.emplace(current_handler_id, std::move(handle));
    return current_handler_id++;
}

bool InputManager::removeHandler(InputManager::HandlerID id)
{
    auto it = handlers.find(id);

    if(it != handlers.end())
    {
        handlers.erase(it);
        return true;
    }

    return false;
}