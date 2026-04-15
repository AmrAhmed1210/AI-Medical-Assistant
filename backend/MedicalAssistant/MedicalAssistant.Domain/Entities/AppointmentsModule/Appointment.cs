using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Domain.Entities.AppointmentsModule
{
    public class Appointment : BaseEntity
    {
        public int PatientId { get; set; }
        public virtual Patient Patient { get; set; } = default!;

        public int DoctorId { get; set; }
        public virtual Doctor Doctor { get; set; } = default!;

        /// <summary>
        /// Date as string e.g. "7 Mar 2026" — matches frontend format
        /// </summary>
        public string Date { get; set; } = string.Empty;

        /// <summary>
        /// Time as string e.g. "10:00 AM"
        /// </summary>
        public string Time { get; set; } = string.Empty;

        /// <summary>
        /// Payment method: "visa" or "cash"
        /// </summary>
        public string PaymentMethod { get; set; } = "cash";

        // Status: Pending, Confirmed, Cancelled
        public string Status { get; set; } = "Pending";

        public string? Notes { get; set; }

        public DateTime CreatedAt { get; set; }
    }
}
