import { useState } from "react"
import { Search, MoreVertical, UserPlus } from "lucide-react"
import { useNavigate } from "react-router-dom"
import "../../styles/dashboard.css"


interface UserEntry {
  id: string
  name: string
  email: string
  role: "Patient" | "Doctor" | "Admin"
  status: "active" | "inactive" | "suspended"
  joinDate: string
  avatar: string
}

const users: UserEntry[] = [
  { id: "1", name: "Mr. Williamson", email: "williamson@mail.com", role: "Patient", status: "active", joinDate: "Jan 2024", avatar: "MW" },
  { id: "2", name: "Dr. Eion Morgan", email: "morgan@medicare.com", role: "Doctor", status: "active", joinDate: "Sep 2023", avatar: "EM" },
  { id: "3", name: "Sarah Mitchell", email: "sarah.m@mail.com", role: "Patient", status: "active", joinDate: "Dec 2023", avatar: "SM" },
  { id: "4", name: "Dr. Chloe Kelly", email: "chloe.k@medicare.com", role: "Doctor", status: "active", joinDate: "Oct 2023", avatar: "CK" },
  { id: "5", name: "Ahmed Hassan", email: "ahmed.h@mail.com", role: "Patient", status: "active", joinDate: "Jan 2024", avatar: "AH" },
  { id: "6", name: "Dr. Lauren Hemp", email: "lauren.h@medicare.com", role: "Doctor", status: "inactive", joinDate: "Nov 2023", avatar: "LH" },
  { id: "7", name: "Emily Watson", email: "emily.w@mail.com", role: "Patient", status: "suspended", joinDate: "Feb 2024", avatar: "EW" },
  { id: "8", name: "Admin Root", email: "admin@medicare.com", role: "Admin", status: "active", joinDate: "Aug 2023", avatar: "AR" },
]

export function AdminUsers() {
    const navigate = useNavigate()
  const [search, setSearch] = useState("")
  const [roleFilter, setRoleFilter] = useState("all")

  const filtered = users.filter((u) => {
    const matchesSearch =
      u.name.toLowerCase().includes(search.toLowerCase()) ||
      u.email.toLowerCase().includes(search.toLowerCase())
    const matchesRole =
      roleFilter === "all" || u.role.toLowerCase() === roleFilter
    return matchesSearch && matchesRole
  })

  return (
    <div className="admin-users">
      <div className="admin-header">
        <div>
          <h1 className="admin-title">Manage Users</h1>
          <p className="admin-subtitle">{users.length} total users</p>
        </div>
        <button className="add-user-btn"   onClick={() => navigate("/add-user")}>
          <UserPlus size={18} />
        </button>
      </div>

      <div className="search-box">
        <Search size={16} />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search users..."
          className="search-input"
        />
      </div>

      <div className="filter-row">
        {["all", "patient", "doctor", "admin"].map((role) => (
          <button
            key={role}
            onClick={() => setRoleFilter(role)}
            className={`filter-btn ${roleFilter === role ? "active" : ""}`}
          >
            {role}
          </button>
        ))}
      </div>

      <div className="users-list">
        {filtered.map((user) => (
          <div key={user.id} className="user-card">
            <div className="avatar">{user.avatar}</div>

            <div className="user-info">
              <div className="user-name-row">
                <span className="user-name">{user.name}</span>
                <span
                  className={`badge role-${user.role.toLowerCase()}`}
                >
                  {user.role}
                </span>
              </div>

              <div className="user-email">{user.email}</div>

              <div className="meta-row">
                <span
                  className={`badge status-${user.status}`}
                >
                  {user.status}
                </span>
                <span className="join-date">
                  Joined {user.joinDate}
                </span>
              </div>
            </div>

            <button className="more-btn">
              <MoreVertical size={18} />
            </button>
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="empty-state">
            No users found
          </div>
        )}
      </div>
    </div>
  )
}