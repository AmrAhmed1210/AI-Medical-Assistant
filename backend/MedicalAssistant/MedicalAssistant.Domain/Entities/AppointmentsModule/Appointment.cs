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

        // Optional split date & time
        public DateTime? AppointmentDate { get; set; }
        public TimeSpan? AppointmentTime { get; set; }

        // Combined scheduled date & time (used by APIs & frontend)
        public DateTime ScheduledAt { get; set; }

        // Linked AI session (optional)
        public int? SessionId { get; set; }

        // Status (Pending, Confirmed, Cancelled, Completed)
        public string Status { get; set; } = "Pending";

        // Cancellation reason
        public string? Reason { get; set; }

        // Doctor completion notes
        public string? Notes { get; set; }

        // Soft delete flag
        public bool IsDeleted { get; set; } = false;

        // Created at
        public DateTime CreatedAt { get; set; }

        // Updated at (status change timestamp)
        public DateTime? UpdatedAt { get; set; }
    }
}
