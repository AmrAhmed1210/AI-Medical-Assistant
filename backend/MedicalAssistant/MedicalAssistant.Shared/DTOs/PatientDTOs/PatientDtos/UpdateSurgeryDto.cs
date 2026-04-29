using System;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs
{
    public class UpdateSurgeryDto
    {
        public string? SurgeryName { get; set; }
        public DateOnly? SurgeryDate { get; set; }
        public string? HospitalName { get; set; }
        public string? DoctorName { get; set; }
        public string? Complications { get; set; }
        public string? Notes { get; set; }
    }
}
