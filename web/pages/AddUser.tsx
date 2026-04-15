import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { Eye, EyeOff, Lock } from "lucide-react"
import "../styles/add-user.css"

export default function AddUser() {
  const navigate = useNavigate()
  const [showPassword, setShowPassword] = useState(false)

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "", // إضافة الحقل هنا
    role: "Patient",
    status: "active",
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log("New User Data:", formData)
    navigate("/users")
  }

  return (
    <div className="add-user-page">
      <div className="add-user-card">
        <div className="card-header">
          <div className="icon-box"><Lock size={20} /></div>
          <h2>Add New User</h2>
        </div>

        <form onSubmit={handleSubmit} className="add-user-form">
          <div className="input-group">
            <input type="text" name="name" placeholder="Full Name" required onChange={handleChange} />
          </div>

          <div className="input-group">
            <input type="email" name="email" placeholder="Email Address" required onChange={handleChange} />
          </div>

          <div className="input-group password-field">
            <input 
              type={showPassword ? "text" : "password"} 
              name="password" 
              placeholder="Password" 
              required 
              onChange={handleChange} 
            />
            <button 
              type="button" 
              className="toggle-password" 
              onClick={() => setShowPassword(!showPassword)}
            >
              {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>

          <div className="select-row">
            <select name="role" onChange={handleChange}>
              <option value="Patient">Patient</option>
              <option value="Doctor">Doctor</option>
              <option value="Admin">Admin</option>
            </select>

            <select name="status" onChange={handleChange}>
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
            </select>
          </div>

          <div className="form-buttons">
            <button type="button" className="cancel-btn" onClick={() => navigate(-1)}>
              Cancel
            </button>
            <button type="submit" className="save-btn">
              Create Account
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}