using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.ReviewsModule;
using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Entities.DoctorsModule;

public class Doctor : BaseEntity
{
    public int UserId { get; set; }
    public User User { get; set; } = null!;

    public int SpecialtyId { get; set; }
    public Specialty Specialty { get; set; } = null!;

    public string Name { get; set; } = string.Empty;
    public string License { get; set; } = string.Empty;
    public string? Bio { get; set; }
    public string? ImageUrl { get; set; }
    public decimal? ConsultationFee { get; set; }
    public int? Experience { get; set; }

    public double Rating { get; set; }
    public int ReviewCount { get; set; }
    public string? Location { get; set; }
    public bool IsAvailable { get; set; } = true;
    public bool IsScheduleVisible { get; set; } = true;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? UpdatedAt { get; set; }

    public ICollection<Review> Reviews { get; set; } = new List<Review>();
    public ICollection<DoctorAvailability> Availabilities { get; set; } = new List<DoctorAvailability>();
    public ICollection<PatientVisit> PatientVisits { get; set; } = new List<PatientVisit>();
}
