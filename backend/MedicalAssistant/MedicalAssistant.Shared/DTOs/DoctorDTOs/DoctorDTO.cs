using System;

namespace MedicalAssistant.Shared.DTOs.DoctorDTOs
{
    public class DoctorDTO
    {
        public int Id { get; set; }

        public string FullName { get; set; } = string.Empty;

        public string Specialty { get; set; } = string.Empty;

        public double Rating { get; set; }

        public int ReviewCount { get; set; }

        public decimal ConsultFee { get; set; }

        public bool IsAvailable { get; set; }

        public string PhotoUrl { get; set; } = string.Empty;

        public string? Location { get; set; }
    }
}
