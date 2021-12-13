#pragma once
#include <pcl/visualization/cloud_viewer.h>
#include <map>
#include "InputHandler.hpp"

namespace clouds
{
    class CloudReg;

    class InputManager
    {
    public:
        using MouseEvent = pcl::visualization::MouseEvent;
        using KeyboardEvent = pcl::visualization::KeyboardEvent;
        using HandlerID = int;

        InputManager();
        
        HandlerID addHandler(InputHandlerBase&&);
        bool removeHandler(HandlerID);

        template<typename T>
        void fireEvent(CloudReg* r, const T& evt)
        {
            for(auto [k, v] : handlers)
            {
                (void)k;
                if(v.suitableFor(evt))
                {
                    v(r, evt);
                }
            }
        }
    private:
        friend class CloudReg;
        static void mouseEvent(const MouseEvent &event, void* reg_ptr);
        static void keyboardEvent(const KeyboardEvent &event, void* reg_ptr);

        void mouseEventImpl(CloudReg*, const MouseEvent&);
        void keyboardEventImpl(CloudReg*, const MouseEvent&);

        std::map<HandlerID, InputHandlerBase&&> handlers;
        int current_handler_id;

    };

}