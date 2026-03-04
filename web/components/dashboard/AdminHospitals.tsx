import { Building2, MapPin, Phone, Plus, Star, Users } from "lucide-react"
import { hospitals } from "../../lib/data"
import "../../styles/hospitals.css"

export function AdminHospitals() {
  return (
    <div className="hospitals-container">
      <div className="hospitals-header">
        <div>
          <h1 className="hospitals-header__title">Hospitals</h1>
          <p style={{ fontSize: '12px', color: '#64748b', margin: '5px 0 0 0' }}>
            {hospitals.length} Managed Medical Facilities
          </p>
        </div>
        <button className="btn-add-hospital" title="Add New Hospital">
          <Plus size={20} />
        </button>
      </div>

      <div className="flex-col-gap">
        {hospitals.map((hospital) => (
          <div
            key={hospital.id}
            className={`hospital-card ${hospital.status !== "active" ? "hospital-card--inactive" : ""}`}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', gap: '15px' }}>
                <div style={{ background: '#eff6ff', padding: '12px', borderRadius: '15px' }}>
                  <Building2 size={24} color="#2563eb" />
                </div>
                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: '800', margin: 0 }}>{hospital.name}</h3>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
                      <Star size={14} fill="#fbbf24" color="#fbbf24" />
                      <span style={{ fontSize: '12px', fontWeight: '700' }}>{hospital.rating}</span>
                    </div>
                    <span style={{ color: '#e2e8f0' }}>|</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#64748b' }}>
                      <Users size={14} />
                      <span style={{ fontSize: '11px', fontWeight: '600' }}>{hospital.doctors.length} Doctors</span>
                    </div>
                  </div>
                </div>
              </div>
              <span className={`status-badge status-badge--${hospital.status}`}>
                {hospital.status}
              </span>
            </div>

            <div className="hospital-contact-box">
              <div className="contact-item">
                <MapPin size={14} color="#2563eb" />
                <span>{hospital.address}</span>
              </div>
              <div className="contact-item">
                <Phone size={14} color="#2563eb" />
                <span>{hospital.phone}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}