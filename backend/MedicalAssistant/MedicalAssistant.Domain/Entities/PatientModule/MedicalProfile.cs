using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class MedicalProfile : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }

        public string? BloodType { get; set; }
        public decimal? WeightKg { get; set; }
        public decimal? HeightCm { get; set; }
        public bool IsSmoker { get; set; }
        public string? SmokingDetails { get; set; }
        public bool DrinksAlcohol { get; set; }
        public string? ExerciseHabits { get; set; }
        public string? EmergencyContactName { get; set; }
        public string? EmergencyContactPhone { get; set; }
        public string? EmergencyContactRelation { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    }
}
