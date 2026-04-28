using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Shared.DTOs.DoctorDTOs
{
    public class DoctorDTO
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public string Specialty { get; set; } = string.Empty;
        public double Rating { get; set; }
        public int ReviewCount { get; set; }
        public string? Location { get; set; }
        public decimal? ConsultationFee { get; set; }
        public bool IsAvailable { get; set; }
        public string ImageUrl { get; set; } = string.Empty;
        public int? YearsExperience { get; set; }
        public bool IsProfileComplete { get; set; }
        public bool IsMobileEnabled { get; set; }
        public bool HasSchedule { get; set; }
        public bool IsScheduleVisible { get; set; } = true;
    }
}
