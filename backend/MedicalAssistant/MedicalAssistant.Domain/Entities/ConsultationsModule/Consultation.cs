using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;

namespace MedicalAssistant.Domain.Entities.ConsultationsModule
{
    public class Consultation : BaseEntity
    {
        public int DoctorId { get; set; }
        public virtual Doctor Doctor { get; set; } = default!;

        public int PatientId { get; set; }
        public virtual Patient Patient { get; set; } = default!;

        /// <summary>
        /// Title of the consultation
        /// </summary>
        public string Title { get; set; } = string.Empty;

        /// <summary>
        /// Detailed description/instructions for the patient
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Scheduled date and time for the consultation
        /// </summary>
        public DateTime ScheduledAt { get; set; }

        /// <summary>
        /// Status: Scheduled, Completed, Cancelled
        /// </summary>
        public string Status { get; set; } = "Scheduled";

        /// <summary>
        /// When the consultation was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// When the consultation was last updated
        /// </summary>
        public DateTime? UpdatedAt { get; set; }
    }
}
