namespace MedicalAssistant.Shared.DTOs.AppointmentsDTOs
{
    public class AppointmentDto
    {
        public int Id { get; set; }
        public int PatientId { get; set; }
        public int DoctorId { get; set; }

        /// <summary>
        /// Doctor name — populated from navigation property
        /// </summary>
        public string DoctorName { get; set; } = string.Empty;

        /// <summary>
        /// Patient name — populated from navigation property
        /// </summary>
        public string PatientName { get; set; } = string.Empty;

        /// <summary>
        /// Doctor specialty name
        /// </summary>
        public string Specialty { get; set; } = string.Empty;

        /// <summary>
        /// Date as string e.g. "7 Mar 2026"
        /// </summary>
        public string Date { get; set; } = string.Empty;

        /// <summary>
        /// Time as string e.g. "10:00 AM"
        /// </summary>
        public string Time { get; set; } = string.Empty;

        /// <summary>
        /// Combined datetime string derived from Date + Time.
        /// </summary>
        public string ScheduledAt { get; set; } = string.Empty;

        /// <summary>
        /// Payment method: "visa" or "cash"
        /// </summary>
        public string PaymentMethod { get; set; } = string.Empty;

        /// <summary>
        /// Status: "confirmed", "pending", "cancelled"
        /// </summary>
        public string Status { get; set; } = string.Empty;

        public string? Notes { get; set; }
        public bool IsFreeRebook { get; set; }
        public bool CanRebook { get; set; }
    }
}
