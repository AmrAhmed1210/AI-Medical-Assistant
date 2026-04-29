using System;

namespace MedicalAssistant.Shared.DTOs.PatientDTOs
{
    public class UpdateMedicalProfileDto
    {
        public string? BloodType { get; set; }
        public decimal? WeightKg { get; set; }
        public decimal? HeightCm { get; set; }
        public bool? IsSmoker { get; set; }
        public string? SmokingDetails { get; set; }
        public bool? DrinksAlcohol { get; set; }
        public string? ExerciseHabits { get; set; }
        public string? EmergencyContactName { get; set; }
        public string? EmergencyContactPhone { get; set; }
        public string? EmergencyContactRelation { get; set; }
    }
}
