using MedicalAssistant.Domain.Entities.DoctorsModule;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class PatientVisit : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }
        public int DoctorId { get; set; }
        public Doctor? Doctor { get; set; }
        public int? AppointmentId { get; set; }

        public string ChiefComplaint { get; set; } = string.Empty;
        public string? PresentIllnessHistory { get; set; }
        public string? ExaminationFindings { get; set; }
        public string? Assessment { get; set; }
        public string? Plan { get; set; }
        public string? Notes { get; set; }
        public string? SummarySnapshot { get; set; }

        public DateOnly VisitDate { get; set; } = DateOnly.FromDateTime(DateTime.UtcNow);
        public string Status { get; set; } = "active";
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? ClosedAt { get; set; }
        public ICollection<Symptom> Symptoms { get; set; } = new List<Symptom>();
        public ICollection<VisitVitalSign> VitalSigns { get; set; } = new List<VisitVitalSign>();
        public ICollection<VisitPrescription> Prescriptions { get; set; } = new List<VisitPrescription>();
        public ICollection<VisitDocument> Documents { get; set; } = new List<VisitDocument>();
    }
}
