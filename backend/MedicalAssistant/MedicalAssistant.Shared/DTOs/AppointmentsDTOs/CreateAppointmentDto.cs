using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.AppointmentsDTOs
{
    public class CreateAppointmentDto
    {
        [Required]
        public int PatientId { get; set; }

        [Required]
        public int DoctorId { get; set; }

        /// <summary>
        /// Date as string e.g. "7 Mar" or "7 Mar 2026"
        /// </summary>
        [Required]
        public string Date { get; set; } = string.Empty;

        /// <summary>
        /// Time as string e.g. "10:00 AM"
        /// </summary>
        [Required]
        public string Time { get; set; } = string.Empty;

        /// <summary>
        /// Payment method: "visa" or "cash"
        /// </summary>
        [Required]
        [RegularExpression("visa|cash", ErrorMessage = "PaymentMethod must be 'visa' or 'cash'")]
        public string PaymentMethod { get; set; } = "cash";

        public string? Notes { get; set; }
    }
}
