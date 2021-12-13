#pragma once
#include <functional>
#include <string>

namespace clouds
{

    class InputManager;
    class CloudReg;

    class InputHandlerBase
    {   
    public:
        template<typename T>
        inline void operator()(CloudReg*, const T& event);

        template<typename T>
        inline bool suitableFor(const T& evt);

        virtual ~InputHandlerBase(){}

    protected:
        friend class InputManager;
        virtual void registerSelf(InputManager*, int) = 0;
    };

    template<typename Event>
    class InputHandler : public InputHandlerBase
    {
    public:
        using Handler = std::function<void(CloudReg*, const Event&)>;

        InputHandler(Handler);
        InputHandler(const std::string&, Handler);
        InputHandler(InputHandler<Event>&) = delete;
        InputHandler(InputHandler<Event>&&);
        virtual ~InputHandler();

        virtual void operator()(CloudReg*, const Event&) const;
    private:
        friend class InputManager;
        
        virtual void registerSelf(InputManager*, int);
        InputManager* parent;
        const std::string name;
        Handler handler;
        int handler_id;
    };

    template<typename T>
    inline bool InputHandlerBase::suitableFor(const T&)
    {
        return dynamic_cast<InputHandler<T>*>(this) != nullptr;
    }

    template<typename T>
    inline void InputHandlerBase::operator()(CloudReg* reg, const T& event)
    {
        auto self = dynamic_cast<InputHandler<T>*>(this);
        if(self)
        {
            self->operator()(reg, event);
        }
    }

    extern template class InputHandler<pcl::visualization::KeyboardEvent>;
    extern template class InputHandler<pcl::visualization::MouseEvent>;
}