using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Domain.Entities.AppointmentsModule
{
    public class Appointment : BaseEntity
    {
        public int PatientId { get; set; }
        public virtual Patient Patient { get; set; } = null!;

        public int DoctorId { get; set; }
        public virtual Doctor Doctor { get; set; } = null!;

        public string Date { get; set; } = string.Empty;

        public string Time { get; set; } = string.Empty;

        public string PaymentMethod { get; set; } = "cash";

        public string Status { get; set; } = "Pending";

        public string? Notes { get; set; }

        public DateTime CreatedAt { get; set; }
    }
}
