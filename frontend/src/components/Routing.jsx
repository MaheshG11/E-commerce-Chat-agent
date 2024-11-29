

import React from 'react'
import { Link,useNavigate} from 'react-router-dom';
import "./Routing.css"
const Routing = () => {
    const navigate = useNavigate();
  return (
    
    <div className='Routing-nav'>
  <button  className="App-button" onClick={() => navigate('/upload_products')}>Upload Product Here</button>
  <button  className="App-button" onClick={() => navigate('/chat')}>Chat Here</button>
</div>
  )
}

export default Routing;