using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Domain.Entities.AppointmentsModule
{
    public class Appointment : BaseEntity
    {
        // Patient who booked the appointment
        public int PatientId { get; set; }
        public virtual Patient Patient { get; set; } = default!;

        // Doctor for the appointment
        public int DoctorId { get; set; }
        public virtual Doctor Doctor { get; set; } = default!;

        // Appointment date and time
        public DateTime AppointmentDate { get; set; }
        public TimeSpan AppointmentTime { get; set; }
        public DateTime ScheduledAt { get; set; }
        // Status (Pending, Confirmed, Cancelled)
        public string Status { get; set; } = "Pending";

        // Optional notes
        public string? Notes { get; set; }

        // Created at
        public DateTime CreatedAt { get; set; }
    }
}
